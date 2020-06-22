import bisect
from datetime import datetime
from typing import Any, Dict, List, Optional

import jax.numpy as np
import pandas as pd

import ergo.distributions as dist

from .types import ArrayLikeType


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
    metaculus: Any
    name: Optional[str]

    def __init__(
        self, id: int, metaculus: Any, data: Dict, name=None,
    ):
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

    @property
    def question_url(self):
        return f"https://{self.metaculus.api_domain}.metaculus.com/questions/{self.id}"

    def __repr__(self):
        if self.name:
            return f'<MetaculusQuestion name="{self.name}">'
        elif self.data:
            return f'<MetaculusQuestion title="{self.title}">'
        else:
            return "<MetaculusQuestion>"

    def __str__(self):
        return repr(self)

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

    def community_dist(self) -> dist.Distribution:
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
