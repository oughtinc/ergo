"""
This module lets you get question and prediction information from Metaculus
and submit predictions, via the API (https://www.metaculus.com/api2/)

**Example**

In this example, we predict the admit rate for Harvard's class of 2029:

https://www.metaculus.com/questions/3622

We predict that the admit rate will be 20% higher than the current community prediction.

.. doctest::
    >>> import os
    >>> import ergo
    >>> import jax.numpy as np
    >>> from numpyro.handlers import seed

    >>> metaculus = ergo.Metaculus(
    ...     username=os.getenv("METACULUS_USERNAME"),
    ...     password=os.getenv("METACULUS_PASSWORD"),
    ...     api_domain="www"
    ... )

    >>> harvard_question = metaculus.get_question(3622)
    >>> # harvard_question.show_community_prediction()

    >>> community_prediction_samples = np.array(
    ...     [harvard_question.sample_community() for _ in range(0, 5000)]
    ... )
    >>> my_prediction_samples = community_prediction_samples * 1.2

    >>> # harvard_question.show_submission(my_prediction_samples)
    >>> harvard_question.submit_from_samples(my_prediction_samples)
    <Response [202]>
"""

import bisect
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import math
import textwrap
from typing import Any, Dict, List, Optional, Union

import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_density,
    geom_histogram,
    ggplot,
    ggtitle,
    guides,
    labs,
    scale_fill_brewer,
    scale_x_continuous,
    scale_x_datetime,
    scale_x_log10,
    theme,
)
import requests
from scipy import stats
from typing_extensions import Literal

from ergo import ppl
from ergo.distributions import Categorical, flip, halfnormal, random_choice
import ergo.logistic as logistic
from ergo.theme import ergo_theme
from ergo.utils import memoized_method

ArrayLikes = [pd.DataFrame, pd.Series, np.ndarray, np.DeviceArray, onp.ndarray]

ArrayLikeType = Union[pd.DataFrame, pd.Series, np.ndarray, np.DeviceArray, onp.ndarray]


@dataclass
class ScoredPrediction:
    """
    A prediction scored according to how it resolved or
    according to the current community prediction
    """

    time: float
    prediction: Any
    resolution: float
    score: float
    question_name: str


class MetaculusQuestion:
    """
    A forecasting question on Metaculus

    :param id: Question id
    :param metaculus: Metaculus API instance
    :param data: Question JSON retrieved from Metaculus API
    :param name: Name to assign to question (used in models)

    :ivar activity:
    :ivar anon_prediction_count:
    :ivar author:
    :ivar author_name:
    :ivar can_use_powers:
    :ivar close_time: when the question closes
    :ivar comment_count:
    :ivar created_time: when the question was created
    :ivar id: question id
    :ivar is_continuous: is the question continuous or binary?
    :ivar last_activity_time:
    :ivar page_url: url for the question page on Metaculus
    :ivar possibilities:
    :ivar prediction_histogram: histogram of the current community prediction
    :ivar prediction_timeseries: predictions on this question over time
    :ivar publish_time: when the question was published
    :ivar resolution:
    :ivar resolve_time: when the question will resolve
    :ivar status:
    :ivar title:
    :ivar type:
    :ivar url:
    :ivar votes:
    """

    id: int
    data: Dict
    metaculus: "Metaculus"
    name: Optional[str]

    def __init__(self, id: int, metaculus: "Metaculus", data: Dict, name=None):
        """
        :param id: question id on Metaculus
        :param metaculus: Metaculus class instance, specifies which user to use for
            e.g. submitting predictions
        :param data: information about the question,
            e.g. as returned by the Metaculus API
        :param name: name for the question to be
            e.g. used in graph titles, defaults to None
        """
        self.id = id
        self.data = data
        self.metaculus = metaculus
        self.name = name

    def __repr__(self):
        if self.name:
            return f'<MetaculusQuestion name="{self.name}">'
        elif self.data:
            return f'<MetaculusQuestion title="{self.title}">'
        else:
            return f"<MetaculusQuestion>"

    def __str__(self):
        return repr(self)
    
    @property
    def latest_community_percentiles(self):
        """
        :return: Some percentiles for the metaculus commununity's latest rough
            prediction. `prediction_histogram` returns a more fine-grained
            histogram of the community prediction
        """
        return self.prediction_timeseries[-1]["community_prediction"]

    def __getattr__(self, name):
        """
        If an attribute isn't directly on the class, check whether it's in the
        raw question data. If it's a time, format it appropriately.

        :param name: attr name
        :return: attr value
        """
        if name in self.data:
            if name.endswith("_time"):
                # could use dateutil.parser to deal with timezones better,
                # but opted for lightweight since datetime.fromisoformat
                # will fix this in python 3.7
                try:
                    # attempt to parse with microseconds
                    return datetime.strptime(self.data[name], "%Y-%m-%dT%H:%M:%S%fZ")
                except ValueError:
                    try:
                        # attempt to parse without microseconds
                        return datetime.strptime(self.data[name], "%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        print(
                            f"The column {name} could not be converted into a datetime"
                        )
                        return self.data[name]

            return self.data[name]
        else:
            raise AttributeError(
                f"Attribute {name} is neither directly on this class nor in the raw question data"
            )

    def set_data(self, key: str, value: Any):
        """
        Set key on data dict

        :param key:
        :param value:
        """
        self.data[key] = value

    def refresh_question(self):
        """
        Refetch the question data from Metaculus,
        used when the question data might have changed
        """
        r = self.metaculus.s.get(f"{self.metaculus.api_url}/questions/{self.id}")
        self.data = r.json()

    def sample_community(self):
        """
        Get one sample from the distribution of the Metaculus community's
        prediction on this question
        (sample is denormalized/on the the true scale of the question)
        """
        raise NotImplementedError("This should be implemented by a subclass")

    @staticmethod
    def to_dataframe(
        questions: List["MetaculusQuestion"],
        columns: List[str] = ["id", "title", "resolve_time"],
    ) -> pd.DataFrame:
        """
        Summarize a list of questions in a dataframe

        :param questions: questions to summarize
        :param columns: list of column names as strings
        :return: pandas dataframe summarizing the questions
        """

        data = [
            [question.name if key == "name" else question.data[key] for key in columns]
            for question in questions
        ]

        return pd.DataFrame(data, columns=columns)

    def get_community_prediction(self, before: datetime = None):
        if len(self.prediction_timeseries) == 0:
            raise LookupError  # No community prediction exists yet

        if before is None:
            return self.prediction_timeseries[-1]["community_prediction"]

        i = bisect.bisect_left(
            [prediction["t"] for prediction in self.prediction_timeseries],
            before.timestamp(),
        )

        if i == len(self.prediction_timeseries):  # No prediction predates
            raise LookupError

        return self.prediction_timeseries[i]["community_prediction"]

    @staticmethod
    def get_central_quantiles(
        df: ArrayLikeType, percent_kept: float = 0.95, side_cut_from: str = "both",
    ):
        """
        Get the values that bound the central (percent_kept) of the sample distribution,
        i.e.,  cutting the tails from these values will give you the central.
        If passed a dataframe with multiple variables, the bounds that encompass
        all variables will be returned.

        :param df: pandas dataframe of one or more column of samples
        :param percent_kept: percentage of sample distrubtion to keep
        :param side_cut_from: which side to cut tails from,
            either 'both','lower', or 'upper'
        :return: lower and upper values of the central (percent_kept) of
            the sample distribution.
        """

        if side_cut_from not in ("both", "lower", "upper"):
            raise ValueError("side keyword must be either 'both','lower', or 'upper'")

        percent_cut = 1 - percent_kept
        if side_cut_from == "lower":
            _lb = percent_cut
            _ub = 1.0
        elif side_cut_from == "upper":
            _lb = 0.0
            _ub = 1 - percent_cut
        else:
            _lb = percent_cut / 2
            _ub = 1 - percent_cut / 2

        if isinstance(df, (pd.Series, np.ndarray)):
            _lq, _uq = df.quantile([_lb, _ub])  # type: ignore
            return (_lq, _uq)

        _lqs = []
        _uqs = []
        for col in df:
            _lq, _uq = df[col].quantile([_lb, _ub])
            _lqs.append(_lq)
            _uqs.append(_uq)
        return (min(_lqs), max(_uqs))


class BinaryQuestion(MetaculusQuestion):
    """
    A binary Metaculus question -- how likely is this event to happen, from 0 to 1?
    """

    def score_prediction(self, prediction, resolution: float) -> ScoredPrediction:
        """
        Score a prediction relative to a resolution using a Brier Score.

        :param prediction: how likely is the event to happen, from 0 to 1?
        :param resolution: how likely is the event to happen, from 0 to 1?
            (0 if it didn't, 1 if it did)
        :return: ScoredPrediction with Brier score, see
            https://en.wikipedia.org/wiki/Brier_score#Definition
            0 is best, 1 is worst, 0.25 is chance
        """
        predicted = prediction["x"]
        score = (resolution - predicted) ** 2
        return ScoredPrediction(
            prediction["t"], prediction, resolution, score, self.__str__()
        )

    def change_since(self, since: datetime):
        """
        Calculate change in community prediction between the argument and most recent
        prediction

        :param since: datetime
        :return: change in community prediction since datetime
        """
        try:
            old = self.get_community_prediction(before=since)
            new = self.get_community_prediction()
        except LookupError:
            # Happens if no prediction predates since or no prediction yet
            return 0

        return new - old

    def score_my_predictions(self):
        """
        Score all of my predictions according to the question resolution
        (or according to the current community prediction if the resolution
        isn't available)

        :return: List of ScoredPredictions with Brier scores
        """
        resolution = self.resolution
        if resolution is None:
            last_community_prediction = self.prediction_timeseries[-1]
            resolution = last_community_prediction["distribution"]["avg"]
        predictions = self.my_predictions["predictions"]
        return [
            self.score_prediction(prediction, resolution) for prediction in predictions
        ]

    def submit(self, p: float) -> requests.Response:
        """
        Submit a prediction to my Metaculus account

        :param p: how likely is the event to happen, from 0 to 1?
        """
        return self.metaculus.post(
            f"{self.metaculus.api_url}/questions/{self.id}/predict/",
            {"prediction": p, "void": False},
        )

    def sample_community(self) -> bool:
        """
        Sample from the Metaculus community distribution (Bernoulli).
        """
        community_prediction = self.get_community_prediction()
        return flip(community_prediction)


@dataclass
class SubmissionLogisticParams(logistic.LogisticParams):
    """
    Parameters needed to submit a logistic to Metaculus as part of a prediction
    """

    low: float
    high: float


@dataclass
class SubmissionMixtureParams:
    """
    Parameters needed to submit a prediction to Metaculus
    (in the form of a logistic mixture)
    """

    components: List[SubmissionLogisticParams]
    probs: List[float]


class ContinuousQuestion(MetaculusQuestion):
    """
    A continuous Metaculus question -- a question of the form,
    what's your distribution on this event?
    """

    @property
    def low_open(self) -> bool:
        """
        Are you allowed to place probability mass below the bottom
        of this question's range?
        """
        return self.possibilities["low"] == "tail"

    @property
    def high_open(self) -> bool:
        """
        Are you allowed to place probability mass
        above the top of this question's range?
        """
        return self.possibilities["high"] == "tail"

    @property
    def has_predictions(self):
        """
        Are there any predictions for the question yet?
        """
        return hasattr(self, "prediction_histogram")

    @property
    def question_range(self):
        """
        Range of answers specified when the question was created
        """
        return self.possibilities["scale"]

    @property
    def question_range_width(self):
        return self.question_range["max"] - self.question_range["min"]

    def _scale_x(self, xmin: float = None, xmax: float = None):
        return scale_x_continuous(limits=(xmin, xmax))

    @property
    def plot_title(self):
        return "\n".join(textwrap.wrap(self.data["title"], 60))  # type: ignore

    # TODO: maybe it's better to fit the logistic first then normalize,
    # rather than the other way around?
    def normalize_samples(self, samples):
        """
        The Metaculus API accepts normalized predictions rather than predictions on
        the true scale of the question. Normalize samples to fit that scale.

        :param samples: samples from the true-scale prediction distribution
        """
        raise NotImplementedError("This should be implemented by a subclass")

    def get_submission_params(
        self, logistic_params: logistic.LogisticParams
    ) -> SubmissionLogisticParams:
        """
        Get the params needed to submit a logistic to Metaculus as part of a prediction.
        See comments for more explanation of how the params need to be transformed
        for Metaculus to accept them

        :param logistic_params: params for a logistic on the normalized scale
        :return: params to submit the logistic to Metaculus as part of a prediction
        """
        distribution = stats.logistic(logistic_params.loc, logistic_params.scale)
        # The loc and scale have to be within a certain range for the
        # Metaculus API to accept the prediction.

        # max loc of 3 set based on API response to prediction on
        # https://pandemic.metaculus.com/questions/3920/what-will-the-cbo-estimate-to-be-the-cost-of-the-emergency-telework-act-s3561/
        max_loc = 3
        clipped_loc = min(logistic_params.loc, max_loc)

        min_scale = 0.01

        # max scale of 10 set based on API response to prediction on
        # https://pandemic.metaculus.com/questions/3920/what-will-the-cbo-estimate-to-be-the-cost-of-the-emergency-telework-act-s3561/
        max_scale = 10
        clipped_scale = min(max(logistic_params.scale, min_scale), max_scale)

        if self.low_open:
            # We're not really sure what the deal with the low and high is.
            # Presumably they're supposed to be the points at which Metaculus "cuts off"
            # your distribution and ignores porbability mass assigned below/above.
            # But we're not actually trying to use them to "cut off" our distribution
            # in a smart way; we're just trying to include as much of our distribution
            # as we can without the API getting unhappy
            # (we belive that if you set the low higher than the value below
            # [or if you set the high lower], then the API will reject the prediction,
            # though we haven't tested that extensively)
            min_open_low = 0.01
            low = max(distribution.cdf(0), min_open_low)
        else:
            low = 0

        if self.high_open:
            # min high of (low + 0.01) set based on API response for
            # https://www.metaculus.com/api2/questions/3961/predict/ --
            # {'prediction': ['high minus low must be at least 0.01']}"
            min_open_high = low + 0.01

            max_open_high = 0.99
            high = max(min(distribution.cdf(1), max_open_high), min_open_high)
        else:
            high = 1

        return SubmissionLogisticParams(clipped_loc, clipped_scale, low, high)

    def denormalize_samples(self, samples) -> np.ndarray:
        """
        Map normalized samples back to the true scale
        """
        raise NotImplementedError("This should be implemented by a subclass")

    @memoized_method(None)
    def community_dist_in_range(self) -> dist.Categorical:
        """
        A distribution for the portion of the current normalized community prediction
        that's within the question's range.

        :return: distribution on integers
        referencing 0...(len(self.prediction_histogram)-1)
        """
        y2 = [p[2] for p in self.prediction_histogram]
        return Categorical(np.array(y2))

    def sample_normalized_community(self) -> float:
        """
        Sample an approximation of the entire current community prediction,
        on the normalized scale. The main reason that it's just an approximation
        is that we don't know exactly where probability mass outside of the question
        range should be, so we place it arbitrarily.

        :return: One sample on the normalized scale
        """

        # FIXME: Samples below/above range are pretty arbitrary
        sample_below_range = -halfnormal(0.1)
        sample_above_range = 1 + halfnormal(0.1)
        sample_in_range = ppl.sample(self.community_dist_in_range()) / float(
            len(self.prediction_histogram)
        )
        p_below = self.latest_community_percentiles["low"]
        p_above = 1 - self.latest_community_percentiles["high"]
        p_in_range = 1 - p_below - p_above
        return float(
            random_choice(
                [sample_below_range, sample_in_range, sample_above_range],
                ps=[p_below, p_in_range, p_above],
            )
        )

    def sample_community(self) -> float:
        """
        Sample an approximation of the entire current community prediction,
        on the true scale of the question.
        The main reason that it's just an approximation is that we don't know
        exactly where probability mass outside of the question range should be,
        so we place it arbitrarily

        :return: One sample on the true scale
        """

        if not self.has_predictions:
            raise ValueError("There are currently no predictions for this question")
        normalized_sample = self.sample_normalized_community()
        sample = np.array(self.denormalize_samples([normalized_sample]))
        if self.name:
            ppl.tag(sample, self.name)
        return float(sample)

    def get_submission(
        self, mixture_params: logistic.LogisticMixtureParams
    ) -> SubmissionMixtureParams:
        """
        Get parameters to submit a prediction to the Metaculus API using a
        logistic mixture

        :param mixture_params: normalized mixture parameters
        :return: normalized parameters clipped and formatted for the API
        """
        submission_logistic_params = [
            self.get_submission_params(logistic_params)
            for logistic_params in mixture_params.components
        ]

        return SubmissionMixtureParams(submission_logistic_params, mixture_params.probs)

    def get_submission_from_samples(
        self, samples: Union[pd.Series, np.ndarray], samples_for_fit=5000, verbose=False
    ) -> SubmissionMixtureParams:
        if not type(samples) in ArrayLikes:
            raise TypeError("Please submit a vector of samples")
        normalized_samples = self.normalize_samples(samples)
        mixture_params = logistic.fit_mixture(
            normalized_samples, num_samples=samples_for_fit, verbose=verbose
        )
        return self.get_submission(mixture_params)

    @staticmethod
    def format_logistic_for_api(
        submission: SubmissionLogisticParams, weight: float
    ) -> dict:
        # convert all the numbers to floats here so that you can be sure that
        # wherever they originated (e.g. numpy), they'll be regular old floats that
        # can be converted to json by json.dumps
        return {
            "kind": "logistic",
            "x0": float(submission.loc),
            "s": float(submission.scale),
            "w": float(weight),
            "low": float(submission.low),
            "high": float(submission.high),
        }

    def submit(self, submission: SubmissionMixtureParams) -> requests.Response:
        prediction_data = {
            "prediction": {
                "kind": "multi",
                "d": [
                    self.format_logistic_for_api(logistic_params, submission.probs[idx])
                    for idx, logistic_params in enumerate(submission.components)
                ],
            },
            "void": False,
        }

        r = self.metaculus.post(
            f"""{self.metaculus.api_url}/questions/{self.id}/predict/""",
            prediction_data,
        )

        self.refresh_question()

        return r

    def submit_from_samples(
        self, samples, samples_for_fit=5000, verbose=False
    ) -> requests.Response:
        """
        Submit prediction to Metaculus based on samples from a prediction distribution

        :param samples: Samples from a distribution answering the prediction question
        :param samples_for_fit: How many samples to take to fit the logistic mixture.
            More will be slower but will give a better fit
        :return: logistic mixture params clipped and formatted to submit to Metaculus
        """
        submission = self.get_submission_from_samples(
            samples, samples_for_fit, verbose=verbose
        )
        return self.submit(submission)

    @staticmethod
    def get_logistic_from_json(logistic_json: Dict) -> SubmissionLogisticParams:
        return SubmissionLogisticParams(
            logistic_json["x0"],
            logistic_json["s"],
            logistic_json["low"],
            logistic_json["high"],
        )

    @classmethod
    def get_submission_from_json(cls, submission_json: Dict) -> SubmissionMixtureParams:
        components = [
            cls.get_logistic_from_json(logistic_json)
            for logistic_json in submission_json
        ]
        probs = [logistic_json["w"] for logistic_json in submission_json]
        return SubmissionMixtureParams(components, probs)

    def get_latest_normalized_prediction(self) -> SubmissionMixtureParams:
        latest_prediction = self.my_predictions["predictions"][-1]["d"]
        return self.get_submission_from_json(latest_prediction)

    def show_prediction(
        self,
        samples,
        plot_samples: bool = True,
        plot_fitted: bool = False,
        percent_kept: float = 0.95,
        side_cut_from: str = "both",
        show_community: bool = False,
        num_samples: int = 1000,
        **kwargs,
    ):
        """
        Plot prediction on the true question scale from samples or a submission
        object. Optionally compare prediction against a sample from the distribution
        of community predictions

        :param samples: samples from a distribution answering the prediction question
            (true scale). Can either be a 1-d array corresponding to one model's
            predictions, or a pandas DataFrame with each column corresponding to
            a distinct model's predictions
        :param plot_samples: boolean indicating whether to plot the raw samples
        :param plot_fitted: boolean indicating whether to compute Logistic Mixture
            Params from samples and plot the resulting fitted distribution. Note
            this is currently only supported for 1-d samples
        :param percent_kept: percentage of sample distrubtion to keep
        :param side_cut_from: which side to cut tails from,
            either 'both','lower', or 'upper'
        :param show_community: boolean indicating whether comparison
            to community predictions should be made
        :param num_samples: number of samples from the community
        :param **kwargs: additional plotting parameters
        """

        df = pd.DataFrame()

        if not plot_fitted and not plot_samples:
            raise ValueError(
                "Nothing to plot. Niether plot_fitted nor plot_samples was True"
            )

        if plot_samples:
            if isinstance(samples, list):
                samples = pd.Series(samples)
            if not type(samples) in ArrayLikes:
                raise ValueError(
                    "Samples should be a list, numpy array or pandas series"
                )
            num_samples = samples.shape[0]

            if type(samples) == pd.DataFrame:
                if plot_fitted and samples.shape[1] > 1:
                    raise ValueError(
                        "For multiple predictions comparisons, only samples can be compared (plot_fitted must be False)"
                    )
                for col in samples:
                    df[col] = self.normalize_samples(samples[col])
            else:
                df["samples"] = self.normalize_samples(samples)

        if plot_fitted:
            prediction = self.get_submission_from_samples(samples)
            df["fitted"] = pd.Series(
                [logistic.sample_mixture(prediction) for _ in range(0, num_samples)]
            )

        if show_community:
            df["community"] = [  # type: ignore
                self.sample_normalized_community() for _ in range(0, num_samples)
            ]

        # get domain for graph given the percentage of distribution kept
        xmin, xmax = self.denormalize_samples(
            self.get_central_quantiles(
                df, percent_kept=percent_kept, side_cut_from=side_cut_from,
            )
        )

        for col in df:
            df[col] = self.denormalize_samples(df[col])

        df = pd.melt(df, var_name="sources", value_name="samples")  # type: ignore

        plot = self.comparison_plot(df, xmin, xmax, **kwargs) + labs(
            x="Prediction",
            y="Density",
            title=self.plot_title + "\n\nPrediction vs Community"
            if show_community
            else self.plot_title,
        )
        try:
            plot.draw()  # type: ignore
        except RuntimeError as err:
            print(err)
            print(
                "The plot was unable to automatically determine a bandwidth. You can manually specify one with the keyword 'bw', e.g., show_prediction(..., bw=.1)"
            )

    def show_community_prediction(
        self,
        percent_kept: float = 0.95,
        side_cut_from: str = "both",
        num_samples: int = 1000,
        **kwargs,
    ):
        """
        Plot samples from the community prediction on this question

        :param percent_kept: percentage of sample distrubtion to keep
        :param side_cut_from: which side to cut tails from,
            either 'both','lower', or 'upper'
        :param num_samples: number of samples from the community
        :param **kwargs: additional plotting parameters
        """
        community_samples = pd.Series(
            [self.sample_normalized_community() for _ in range(0, num_samples)]
        )

        _xmin, _xmax = self.denormalize_samples(
            self.get_central_quantiles(
                community_samples,
                percent_kept=percent_kept,
                side_cut_from=side_cut_from,
            )
        )

        df = pd.DataFrame(data={"samples": self.denormalize_samples(community_samples)})

        plot = self.density_plot(df, _xmin, _xmax, **kwargs) + labs(
            x="Prediction",
            y="Density",
            title=self.plot_title + "\n\nCommunity Predictions",
        )
        try:
            plot.draw()  # type: ignore
        except RuntimeError as err:
            print(err)
            print(
                "The plot was unable to automatically determine a bandwidth. You can manually specify one with the keyword 'bw', e.g., show_prediction(..., bw=.1)"
            )

    def comparison_plot(
        self, df: pd.DataFrame, xmin=None, xmax=None, bw="normal_reference", **kwargs
    ):
        return (
            ggplot(df, aes(df.columns[1], fill=df.columns[0]))
            + scale_fill_brewer(type="qual", palette="Pastel1")
            + geom_density(bw=bw, alpha=0.8)
            + ggtitle(self.plot_title)
            + self._scale_x(xmin, xmax)
            + ergo_theme
        )

    def density_plot(
        self,
        df: pd.DataFrame,
        xmin=None,
        xmax=None,
        fill: str = "#fbb4ae",
        bw="normal_reference",
        **kwargs,
    ):
        return (
            ggplot(df, aes(df.columns[0]))
            + geom_density(fill=fill, alpha=0.8)
            + ggtitle(self.plot_title)
            + self._scale_x(xmin, xmax)
            + ergo_theme
        )

    def change_since(self, since: datetime):
        """
        Calculate change in community prediction median between the argument and most
        recent prediction

        :param since: datetime
        :return: change in median community prediction since datetime
        """
        try:
            old = self.get_community_prediction(before=since)
            new = self.get_community_prediction()
        except LookupError:
            return 0

        return new["q2"] - old["q2"]


class LinearQuestion(ContinuousQuestion):
    """
    A continuous Metaculus question that's on a linear (as opposed to a log) scale"
    """

    def normalize_samples(self, samples):
        """
        Map samples from their true scale to the Metaculus normalized scale

        :param samples: samples from a distribution answering the prediction question
            (true scale)
        :return: samples on the normalized scale
        """
        return (samples - self.question_range["min"]) / (self.question_range_width)

    def denormalize_samples(self, samples):
        """
        Map samples from the Metaculus normalized scale to the true scale

        :param samples: samples on the normalized scale
        :return: samples from a distribution answering the prediction question
            (true scale)
        """

        # in case samples are in some other array-like format
        samples = np.array(samples)
        return self.question_range["min"] + (self.question_range_width) * samples

    # TODO: also return low and high on the true scale,
    # and use those somehow in logistic.py
    def get_true_scale_logistic_params(
        self, submission_logistic_params: SubmissionLogisticParams
    ) -> logistic.LogisticParams:
        """
        Get logistic params on the true scale of the question,
        from submission normalized params

        :param submission_logistic_params: normalized params
        :return: params on the true scale of the question
        """

        true_loc = (
            submission_logistic_params.loc * self.question_range_width
            + self.question_range["min"]
        )

        true_scale = submission_logistic_params.scale * self.question_range_width

        return logistic.LogisticParams(true_loc, true_scale)

    def get_true_scale_mixture(
        self, submission_params: SubmissionMixtureParams
    ) -> logistic.LogisticMixtureParams:
        """
        Get logistic mixture params on the true scale of the question,
        from normalized submission params

        :param submission_params: params formatted for submission to Metaculus
        :return: params on the true scale of the question
        """
        true_scale_logistics_params = [
            self.get_true_scale_logistic_params(submission_logistic_params)
            for submission_logistic_params in submission_params.components
        ]

        return logistic.LogisticMixtureParams(
            true_scale_logistics_params, submission_params.probs
        )


class LogQuestion(ContinuousQuestion):
    @property
    def deriv_ratio(self) -> float:
        return self.possibilities["scale"]["deriv_ratio"]

    def _scale_x(self, xmin: float = None, xmax: float = None):
        return scale_x_log10(limits=(xmin, xmax))

    def normalized_from_true_value(self, true_value) -> float:
        """
        Get a prediciton sample value on the normalized scale from a true-scale value

        :param true_value: a sample value on the true scale
        :return: a sample value on the normalized scale
        """
        shifted = true_value - self.question_range["min"]
        numerator = shifted * (self.deriv_ratio - 1)
        scaled = numerator / self.question_range_width
        timber = 1 + scaled
        floored_timber = max(timber, 1e-9)
        return math.log(floored_timber, self.deriv_ratio)

    def true_from_normalized_value(self, normalized_value):
        """
        Get a prediciton sample value on the true scale from a normalized-scale value

        :param normalized_value: [description]
        :type normalized_value: [type]
        :return: [description]
        :rtype: [type]
        """
        deriv_term = (self.deriv_ratio ** normalized_value - 1) / (self.deriv_ratio - 1)
        scaled = self.question_range_width * deriv_term
        return self.question_range["min"] + scaled

    def normalize_samples(self, samples):
        """
        Map samples from the true scale to the normalized scale

        :param samples: Samples on the true scale
        :return: Samples on the normalized scale
        """
        return [self.normalized_from_true_value(sample) for sample in samples]

    def denormalize_samples(self, samples):
        """
        Map samples from the normalized scale to the true scale

        :param samples: Samples on the normalized scale
        :return: Samples on the true scale
        """
        return [self.true_from_normalized_value(sample) for sample in samples]


class LinearDateQuestion(LinearQuestion):
    # TODO: add log functionality (if some psychopath makes a log scaled date question)

    def _scale_x(self, xmin: float = None, xmax: float = None):
        return scale_x_datetime(limits=(xmin, xmax))

    @property
    def question_range(self):
        """
        Question range from the Metaculus data plus the question's data range
        """
        qr = {
            "min": 0,
            "max": 1,
            "date_min": datetime.strptime(
                self.possibilities["scale"]["min"], "%Y-%m-%d"
            ).date(),
            "date_max": datetime.strptime(
                self.possibilities["scale"]["max"], "%Y-%m-%d"
            ).date(),
        }
        qr["date_range"] = (qr["date_max"] - qr["date_min"]).days
        return qr

    # TODO Make less fancy. Would be better to only accept datetimes
    def normalize_samples(self, samples):
        """
        Normalize samples from dates to the normalized scale used by the Metaculus API

        :param samples: dates from the predicted distribution answering the question
        :return: normalized samples
        """
        if isinstance(samples[0], date):
            if type(samples) != pd.Series:
                try:
                    samples = pd.Series(samples)
                except ValueError:
                    raise ValueError("Could not process samples vector")
            return self.normalize_dates(samples)
        else:
            return super().normalize_samples(samples)

    def normalize_dates(self, dates: pd.Series):
        """
        Map dates to the normalized scale used by the Metaculus API

        :param dates: a pandas series of dates
        :return: normalized samples
        """

        return (dates - self.question_range["date_min"]).dt.days / self.question_range[
            "date_range"
        ]

    def denormalize_samples(self, samples):
        """
        Map normalized samples to dates using the date range from the question

        :param samples: normalized samples
        :return: dates
        """

        def denorm(sample):
            return self.question_range["date_min"] + timedelta(
                days=round(self.question_range["date_range"] * sample)
            )

        if type(samples) == float:
            return denorm(samples)
        else:
            samples = pd.Series(samples)
            return samples.apply(denorm)

    # TODO enforce return type date/datetime
    def sample_community(self):
        """
        Sample an approximation of the entire current community prediction,
        on the true scale of the question.

        :return: One sample on the true scale
        """
        normalized_sample = self.sample_normalized_community()
        return self.denormalize_samples(normalized_sample)

    def comparison_plot(  # type: ignore
        self, df: pd.DataFrame, xmin=None, xmax=None, bins: int = 50, **kwargs
    ):

        return (
            ggplot(df, aes(df.columns[1], fill=df.columns[0]))
            + scale_fill_brewer(type="qual", palette="Pastel1")
            + geom_histogram(position="identity", alpha=0.9, bins=bins)
            + self._scale_x(xmin, xmax)
            + facet_wrap(df.columns[0], ncol=1)
            + guides(fill=False)
            + ergo_theme
            + theme(axis_text_x=element_text(rotation=45, hjust=1))
        )

    def density_plot(  # type: ignore
        self,
        df: pd.DataFrame,
        xmin=None,
        xmax=None,
        fill: str = "#fbb4ae",
        bins: int = 50,
        **kwargs,
    ):

        return (
            ggplot(df, aes(df.columns[0]))
            + geom_histogram(fill=fill, bins=bins)
            + self._scale_x(xmin, xmax)
            + ergo_theme
            + theme(axis_text_x=element_text(rotation=45, hjust=1))
        )


class Metaculus:
    """
    The main class for interacting with Metaculus

    :param username: A Metaculus username
    :param password: The password for the given Metaculus username
    :param api_domain: A Metaculus subdomain (e.g., www, pandemic, finance)
    """

    player_status_to_api_wording = {
        "predicted": "guessed_by",
        "not-predicted": "not_guessed_by",
        "author": "author",
        "interested": "upvoted_by",
    }

    def __init__(self, username: str, password: str, api_domain: str = "www"):
        self.user_id = None
        self.api_url = f"https://{api_domain}.metaculus.com/api2"
        self.s = requests.Session()
        self.login(username, password)

    def login(self, username, password):
        """
        log in to Metaculus using your credentials and store cookies,
        etc. in the session object for future use
        """
        loginURL = f"{self.api_url}/accounts/login/"
        r = self.s.post(
            loginURL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"username": username, "password": password}),
        )

        self.user_id = r.json()["user_id"]

    def post(self, url: str, data: Dict) -> requests.Response:
        """
        Make a post request using your Metaculus credentials.
        Best to use this for all post requests to avoid auth issues
        """
        r = self.s.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Referer": self.api_url,
                "X-CSRFToken": self.s.cookies.get_dict()["csrftoken"],
            },
            data=json.dumps(data),
        )
        try:
            r.raise_for_status()

        except requests.exceptions.HTTPError as e:
            e.args = (
                str(e.args),
                f"request body: {e.request.body}",
                f"response json: {e.response.json()}",
            )
            raise

        return r

    def make_question_from_data(self, data: Dict, name=None) -> MetaculusQuestion:
        """
        Make a MetaculusQuestion given data about the question
        of the sort returned by the Metaculus API.

        :param data: the question data (usually from the Metaculus API)
        :param name: a custom name for the question
        :return: A MetaculusQuestion from the appropriate subclass
        """
        if not name:
            name = data.get("title")
        if data["possibilities"]["type"] == "binary":
            return BinaryQuestion(data["id"], self, data, name)
        if data["possibilities"]["type"] == "continuous":
            if data["possibilities"]["scale"]["deriv_ratio"] != 1:
                if data["possibilities"].get("format") == "date":
                    raise NotImplementedError(
                        "Logarithmic date-valued questions are not currently supported"
                    )
                else:
                    return LogQuestion(data["id"], self, data, name)
            if data["possibilities"].get("format") == "date":
                return LinearDateQuestion(data["id"], self, data, name)
            else:
                return LinearQuestion(data["id"], self, data, name)
        raise NotImplementedError(
            "We couldn't determine whether this question was binary, continuous, or something else"
        )

    def get_question(self, id: int, name=None) -> MetaculusQuestion:
        """
        Load a question from Metaculus

        :param id: Question id (can be read off from URL)
        :param name: Name to assign to this question (used in models)
        """
        r = self.s.get(f"{self.api_url}/questions/{id}")
        data = r.json()
        if not data.get("possibilities"):
            raise ValueError(
                "Unable to find a question with that id. Are you using the right api_domain?"
            )
        return self.make_question_from_data(data, name)

    def get_questions(
        self,
        question_status: Literal[
            "all", "upcoming", "open", "closed", "resolved", "discussion"
        ] = "all",
        player_status: Literal[
            "any", "predicted", "not-predicted", "author", "interested", "private"
        ] = "any",  # 20 results per page
        cat: Union[str, None] = None,
        pages: int = 1,
    ) -> List["MetaculusQuestion"]:
        """
        Retrieve multiple questions from Metaculus API.

        :param question_status: Question status
        :param player_status: Player's status on this question
        :param cat: Category slug
        :param pages: Number of pages of questions to retrieve
        """

        questions_json = self.get_questions_json(
            question_status, player_status, cat, pages, False
        )
        return [self.make_question_from_data(q) for q in questions_json]

    def get_questions_json(
        self,
        question_status: Literal[
            "all", "upcoming", "open", "closed", "resolved", "discussion"
        ] = "all",
        player_status: Literal[
            "any", "predicted", "not-predicted", "author", "interested", "private"
        ] = "any",  # 20 results per page
        cat: Union[str, None] = None,
        pages: int = 1,
        include_discussion_questions: bool = False,
    ) -> List[Dict]:
        """
        Retrieve JSON for multiple questions from Metaculus API.

        :param question_status: Question status
        :param player_status: Player's status on this question
        :param cat: Category slug
        :param pages: Number of pages of questions to retrieve
        :include_discussion_questions: If true, data for non-prediction
            questions will be included
        """
        query_params = [f"status={question_status}", "order_by=-publish_time"]
        if player_status != "any":
            if player_status == "private":
                query_params.append("access=private")
            else:
                query_params.append(
                    f"{self.player_status_to_api_wording[player_status]}={self.user_id}"
                )

        if cat is not None:
            query_params.append(f"search=cat:{cat}")

        query_string = "&".join(query_params)

        def get_questions_for_pages(
            query_string: str, max_pages: int = 1, current_page: int = 1, results=[]
        ) -> List[Dict]:
            if current_page > max_pages:
                return results

            r = self.s.get(
                f"{self.api_url}/questions/?{query_string}&page={current_page}"
            )

            if r.json() == {"detail": "Invalid page."}:
                return results

            r.raise_for_status()

            return get_questions_for_pages(
                query_string, max_pages, current_page + 1, results + r.json()["results"]
            )

        questions = get_questions_for_pages(query_string, pages)

        # add additional fields ommited from previous query
        for i, q in enumerate(questions):
            r = self.s.get(f"{self.api_url}/questions/{q['id']}")
            questions[i] = dict(r.json(), **q)

        if not include_discussion_questions:
            questions = [
                q for q in questions if q["possibilities"]["type"] != "discussion"
            ]

        return questions

    def make_questions_df(
        self, questions_json: List[Dict], columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert JSON returned by Metaculus API to dataframe.

        :param questions_json: List of questions (as dicts)
        :param columns: Optional list of column names to include
            (if omitted, every column is included)
        """
        if columns is not None:
            questions_df = pd.DataFrame(
                [{k: v for (k, v) in q.items() if k in columns} for q in questions_json]
            )
        else:
            questions_df = pd.DataFrame(questions_json)

        for col in ["created_time", "publish_time", "close_time", "resolve_time"]:
            if col in questions_df.columns:
                questions_df[col] = questions_df[col].apply(
                    lambda x: datetime.strptime(x[:19], "%Y-%m-%dT%H:%M:%S")
                )

        if "author" in questions_df.columns:
            questions_df["i_created"] = questions_df["author"] == self.user_id

        if "my_predictions" in questions_df.columns:
            questions_df["i_predicted"] = questions_df["my_predictions"].apply(
                lambda x: x is not None
            )

        return questions_df
