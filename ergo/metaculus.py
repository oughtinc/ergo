from dataclasses import dataclass
import functools
import json
import math
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import pendulum
import pyro.distributions as dist
import requests
from scipy import stats
import seaborn
import torch
from typing_extensions import Literal

import ergo.logistic as logistic
import ergo.ppl as ppl


@dataclass
class ScoredPrediction:
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
    :ivar close_time:
    :ivar comment_count:
    :ivar created_time:
    :ivar id:
    :ivar is_continuous:
    :ivar last_activity_time:
    :ivar latest_community_prediction:
    :ivar page_url:
    :ivar possibilities:
    :ivar prediction_histogram:
    :ivar prediction_timeseries:
    :ivar publish_time:
    :ivar resolution:
    :ivar resolve_time:
    :ivar status:
    :ivar title:
    :ivar type:
    :ivar url:
    :ivar votes:
    """

    id: int
    data: Optional[object]
    metaculus: "Metaculus"
    name: Optional[str]

    def __init__(self, id: int, metaculus: "Metaculus", data, name=None):
        self.id = id
        self.data = data
        self.metaculus = metaculus
        self.name = name

    @property
    def is_continuous(self) -> bool:
        return self.type == "continuous"

    @property
    def type(self) -> str:
        return self.possibilities["type"]

    @property
    def latest_community_prediction(self):
        return self.prediction_timeseries[-1]["community_prediction"]

    def __getattr__(self, name):
        if name in self.data:
            if name.endswith("_time"):
                return pendulum.parse(self.data[name])  # TZ
            return self.data[name]
        else:
            raise AttributeError(name)

    def __str__(self):
        if self.data:
            return self.data["title"]
        return "<MetaculusQuestion>"

    def refresh_question(self):
        """
        Reload question data from Metaculus
        """
        r = self.metaculus.s.get(f"{self.metaculus.api_url}/questions/{self.id}")
        self.data = r.json()

    @staticmethod
    def to_dataframe(questions: List["MetaculusQuestion"]) -> pd.DataFrame:
        show_names = any(q.name for q in questions)
        if show_names:
            columns = ["id", "name", "title", "resolve_time"]
            data = [
                [question.id, question.name, question.title, question.resolve_time]
                for question in questions
            ]
        else:
            columns = ["id", "title", "resolve_time"]
            data = [
                [question.id, question.title, question.resolve_time]
                for question in questions
            ]
        return pd.DataFrame(data, columns=columns)

    def sample_community(self):
        """
        Sample from community distribution
        """
        raise NotImplementedError("This should be implemented by a subclass")


class BinaryQuestion(MetaculusQuestion):
    def score_prediction(self, prediction, resolution) -> ScoredPrediction:
        predicted = prediction["x"]
        # Brier score, see https://en.wikipedia.org/wiki/Brier_score#Definition. 0 is best, 1 is worst, 0.25 is chance
        score = (resolution - predicted) ** 2
        return ScoredPrediction(
            prediction["t"], prediction, resolution, score, self.__str__()
        )

    def get_scored_predictions(self):
        resolution = self.resolution
        if resolution is None:
            last_community_prediction = self.prediction_timeseries[-1]
            resolution = last_community_prediction["distribution"]["avg"]
        predictions = self.my_predictions["predictions"]
        return [
            self.score_prediction(prediction, resolution) for prediction in predictions
        ]

    def submit(self, p: float):
        return self.metaculus.post(
            f"{self.metaculus.api_url}/questions/{self.id}/predict/",
            {"prediction": p, "void": False},
        )


@dataclass
class SubmissionLogisticParams(logistic.LogisticParams):
    low: float
    high: float


@dataclass
class SubmissionMixtureParams:
    components: List[SubmissionLogisticParams]
    probs: List[float]


class ContinuousQuestion(MetaculusQuestion):
    @property
    def low_open(self) -> bool:
        return self.possibilities["low"] == "tail"

    @property
    def high_open(self) -> bool:
        return self.possibilities["high"] == "tail"

    @property
    def question_range(self):
        return self.possibilities["scale"]

    @property
    def question_range_width(self):
        return self.question_range["max"] - self.question_range["min"]

    def normalize_samples(self, samples):
        """The Metaculus API accepts normalized predictions rather than predictions on the actual scale of the question
        TODO: maybe it's better to fit the logistic first then normalize, rather than the other way around?"""
        raise NotImplementedError("This should be implemented by a subclass")

    def get_submission_params(
        self, logistic_params: logistic.LogisticParams
    ) -> SubmissionLogisticParams:
        distribution = stats.logistic(logistic_params.loc, logistic_params.scale)
        # The loc and scale have to be within a certain range for the Metaculus API to accept the prediction.

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
            # Presumably they're supposed to be the points at which Metaculus "cuts off" your distribution
            # and ignores porbability mass assigned below/above.
            # But we're not actually trying to use them to "cut off" our distribution in a smart way;
            # we're just trying to include as much of our distribution as we can without the API getting unhappy
            # (we belive that if you set the low higher than the value below [or if you set the high lower],
            # then the API will reject the prediction, though we haven't tested that extensively)
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
        """Map normalized samples back to actual scale"""
        raise NotImplementedError("This should be implemented by a subclass")

    @functools.lru_cache(None)
    def community_dist_in_range(self):
        """distribution on integers referencing 0...(len(self.prediction_histogram)-1)"""
        y2 = [p[2] for p in self.prediction_histogram]
        return dist.Categorical(probs=torch.Tensor(y2))

    def sample_normalized_community(self):
        # 0.02 is chosen pretty arbitrarily based on what I think will work best
        # from playing around with the Metaculus API previously.
        # Feel free to tweak if something else seems better.
        # Ideally this wouldn't be a fixed number and would depend on
        # how spread out we actually expect the probability mass to be
        outside_range_scale = 0.02

        sample_below_range = -abs(np.random.logistic(0, outside_range_scale))
        sample_above_range = abs(np.random.logistic(1, outside_range_scale))
        sample_in_range = ppl.sample(self.community_dist_in_range()) / float(
            len(self.prediction_histogram)
        )

        p_below = self.latest_community_prediction["low"]
        p_above = 1 - self.latest_community_prediction["high"]
        p_in_range = 1 - p_below - p_above

        return ppl.random_choice(
            [sample_below_range, sample_in_range, sample_above_range],
            ps=[p_below, p_in_range, p_above],
        )

    def sample_community(self):
        normalized_sample = self.sample_normalized_community()
        sample = torch.Tensor(self.denormalize_samples([normalized_sample]))
        if self.name:
            ppl.tag(sample, self.name)
        return sample

    def get_submission(
        self, mixture_params: logistic.LogisticMixtureParams
    ) -> SubmissionMixtureParams:
        submission_logistic_params = [
            self.get_submission_params(logistic_params)
            for logistic_params in mixture_params.components
        ]

        return SubmissionMixtureParams(submission_logistic_params, mixture_params.probs)

    def get_submission_from_samples(
        self, samples, samples_for_fit=5000
    ) -> SubmissionMixtureParams:
        normalized_samples = self.normalize_samples(samples)
        mixture_params = logistic.fit_mixture(
            normalized_samples, num_samples=samples_for_fit
        )
        return self.get_submission(mixture_params)

    @staticmethod
    def format_logistic_for_api(
        submission: SubmissionLogisticParams, weight: float
    ) -> dict:
        # convert all the numbers to floats here so that you can be sure that wherever they originated
        # (e.g. numpy), they'll be regular old floats that can be converted to json by json.dumps
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

    def submit_from_samples(self, samples, samples_for_fit=5000) -> requests.Response:
        submission = self.get_submission_from_samples(samples, samples_for_fit)
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

    def show_performance(self):
        """TODO: show vs. Metaculus and vs. resolution if available"""
        mixture_params = self.get_latest_normalized_prediction()
        self.show_prediction(mixture_params)

    def show_submission(self, samples):
        raise NotImplementedError("This should be implemented by a subclass")

    def show_community_prediction(self):
        raise NotImplementedError("This should be implemented by a subclass")

    def show_prediction(self, prediction: SubmissionMixtureParams):
        raise NotImplementedError("This should be implemented by a subclass")


class LinearQuestion(ContinuousQuestion):
    def normalize_samples(self, samples):
        return (samples - self.question_range["min"]) / (self.question_range_width)

    def denormalize_samples(self, samples):
        # in case samples are in some other array-like format
        samples = np.array(samples)
        return self.question_range["min"] + (self.question_range_width) * samples

    def get_true_scale_logistic_params(
        self, submission_logistic_params: SubmissionLogisticParams
    ) -> logistic.LogisticParams:
        """Get the logistic on the actual scale of the question,
        from the normalized logistic used in the submission
        TODO: also return low and high on the true scale"""

        true_loc = (
            submission_logistic_params.loc * self.question_range_width
            + self.question_range["min"]
        )

        true_scale = submission_logistic_params.scale * self.question_range_width

        return logistic.LogisticParams(true_loc, true_scale)

    def get_true_scale_mixture(
        self, submission_params: SubmissionMixtureParams
    ) -> logistic.LogisticMixtureParams:
        true_scale_logistics_params = [
            self.get_true_scale_logistic_params(submission_logistic_params)
            for submission_logistic_params in submission_params.components
        ]

        return logistic.LogisticMixtureParams(
            true_scale_logistics_params, submission_params.probs
        )

    def show_prediction(self, prediction: SubmissionMixtureParams, samples=None):
        true_scale_prediction = self.get_true_scale_mixture(prediction)

        pyplot.figure()
        pyplot.title(f"{self} prediction", y=1.07)  # type: ignore
        logistic.plot_mixture(
            true_scale_prediction,  # type: ignore
            data=samples,
        )  # type: ignore

    def show_submission(self, samples):
        submission = self.get_submission_from_samples(samples)

        self.show_prediction(submission, samples)

    def show_community_prediction(self, only_show_this=True):
        if only_show_this:
            pyplot.figure()
            pyplot.title(f"{self} community prediction", y=1.07)

        community_samples = [self.sample_community() for _ in range(0, 1000)]

        ax = seaborn.distplot(community_samples, label="Community")

        pyplot.legend()

        if only_show_this:
            pyplot.show()

        return ax


class LogQuestion(ContinuousQuestion):
    @property
    def deriv_ratio(self) -> float:
        return self.possibilities["scale"]["deriv_ratio"]

    def normalized_from_true_value(self, true_value) -> float:
        shifted = true_value - self.question_range["min"]
        numerator = shifted * (self.deriv_ratio - 1)
        scaled = numerator / self.question_range_width
        timber = 1 + scaled
        floored_timber = max(timber, 1e-9)
        return math.log(floored_timber, self.deriv_ratio)

    def true_from_normalized_value(self, normalized_value):
        deriv_term = (self.deriv_ratio ** normalized_value - 1) / (self.deriv_ratio - 1)
        scaled = self.question_range_width * deriv_term
        return self.question_range["min"] + scaled

    def normalize_samples(self, samples):
        return [self.normalized_from_true_value(sample) for sample in samples]

    def denormalize_samples(self, samples):
        return [self.true_from_normalized_value(sample) for sample in samples]

    @staticmethod
    def set_true_x_ticks():
        true_tick_values = [
            f"{np.exp(log_tick):.1e}" for log_tick in pyplot.xticks()[0]
        ]

        pyplot.xticks(pyplot.xticks()[0], true_tick_values, rotation="vertical")

    def plot_log_prediction(self, prediction: SubmissionMixtureParams, samples=None):
        prediction_normed_samples = np.array(
            [logistic.sample_mixture(prediction) for _ in range(0, 5000)]
        )

        prediction_true_scale_samples = np.array(
            [
                self.true_from_normalized_value(submission_sample)
                for submission_sample in prediction_normed_samples
            ]
        )

        ax = seaborn.distplot(np.log(prediction_true_scale_samples), label="Mixture")
        ax.set(xlabel="Sample value", ylabel="Density")

        if samples is not None:
            seaborn.distplot(np.log(samples), label="Data")

        pyplot.legend()  # type: ignore

    def plot_log_community_prediction(self):
        community_samples = np.log([self.sample_community() for _ in range(0, 1000)])

        ax = seaborn.distplot(community_samples, label="Community")

        pyplot.legend()

        return ax

    def show_submission(self, samples, show_community=False):
        submission = self.get_submission_from_samples(samples)

        self.show_prediction(submission, samples, show_community)

    def show_community_prediction(self):
        pyplot.figure()
        pyplot.title(f"{self} community prediction", y=1.07)
        self.plot_log_community_prediction()
        self.set_true_x_ticks()
        pyplot.show()

    def show_prediction(
        self, prediction: SubmissionMixtureParams, samples=None, show_community=False
    ):
        pyplot.figure()
        pyplot.title(f"{self} prediction", y=1.07)  # type: ignore
        self.plot_log_prediction(prediction, samples)

        if show_community:
            self.plot_log_community_prediction()

        self.set_true_x_ticks()
        pyplot.show()


class ContinuousDateQuestion(ContinuousQuestion):
    # TODO: add log functionality (if some psychopath makes a log date question)

    @property
    def question_range(self):
        return {
            "min": 0,
            "max": 1,
            "date_min": pendulum.parse(self.possibilities["scale"]["min"], exact=True),
            "date_max": pendulum.parse(self.possibilities["scale"]["max"], exact=True)
        }

    # Make sure we are dealing with pendulum dates
    # TODO use @functools.singledispatchmethod to provide functionality for single dates
    def ensure_date_format(self, data):
        print(type(data))
        if type(data) != pd.Series:
            raise ValueError(
                "Was expecting a vector of Dates")
        elif type(data[0]) == pendulum.Date:
            return data
        else:
            try:
                return data.apply(lambda x: pendulum.parse(x, exact=True))
            except:
                raise ValueError(
                    "The samples need to be convertable dates for this question")

    # User helper-function that goes from Dates -> Float Normalized wrt Question Range (as accepted and produced by the Metaculus API)
    # Assumes pd.Series of Dates
    # TODO use @functools.singledispatchmethod to provide functionality for single dates
    def normalize_dates(self, dates):
        if self.is_log:
            raise NotImplementedError(
                "Scaling the normalized prediction to the true scale from the question not yet implemented for questions on the log scale")
        dates = self.ensure_date_format(dates)

        def normalize(date: pendulum.Date):
            return (date - self.question_range["date_min"]).in_days() / (self.question_range["date_max"] - self.question_range["date_min"]).in_days()
        return dates.apply(lambda x: normalize(x))

    # Map normalized samples back to question date scale
    def denormalize_samples_to_date_scale(self, samples):
        if self.is_log:
            raise NotImplementedError(
                "Scaling the normalized prediction to the true scale from the question not yet implemented for questions on the log scale")
        return self.question_range["date_min"].add(days=round((self.question_range["date_max"] - self.question_range["date_min"]).in_days() * samples))


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
        loginURL = f"{self.api_url}/accounts/login/"
        r = self.s.post(
            loginURL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"username": username, "password": password}),
        )

        self.user_id = r.json()["user_id"]

    def post(self, url: str, data: Dict):
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

    def make_question_from_data(self, data=Dict, name=None) -> MetaculusQuestion:
        if data["possibilities"]["type"] == "binary":
            return BinaryQuestion(data["id"], self, data, name)
        if data["possibilities"]["type"] == "continuous":
            if data["possibilities"]["scale"]["deriv_ratio"] != 1:
                if(data["possibilities"]["format"] == "date"):
                    raise NotImplementedError(
                        "Support for logarithmic date-valued questions is not currently supported"
                    )
                else:
                    return LogQuestion(data["id"], self, data, name)
            if(data["possibilities"]["format"] == "date"):
                return ContinuousDateQuestion(data["id"], self, data, name)
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
                "There was not a question with that id. HINT are you using the right api_domain?")
        return self.make_question_from_data(data, name)

    def get_questions_json(
        self,
        question_status: Literal[
            "all", "upcoming", "open", "closed", "resolved", "discussion"
        ] = "all",
        player_status: Literal[
            "any", "predicted", "not-predicted", "author", "interested", "private"
        ] = "any",  # 20 results per page
        pages: int = 1,
    ) -> List[Dict]:
        """
        Retrieve JSON for multiple questions from Metaculus API.

        :param question_status: Question status
        :param player_status: Player's status on this question
        :param pages: Number of pages of questions to retrieve
        """
        query_params = [f"status={question_status}", "order_by=-publish_time"]
        if player_status != "any":
            if player_status == "private":
                query_params.append("access=private")
            else:
                query_params.append(
                    f"{self.player_status_to_api_wording[player_status]}={self.user_id}"
                )

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

        return get_questions_for_pages(query_string, pages)

    def make_questions_df(self, questions_json: List[Dict]) -> pd.DataFrame:
        """
        Convert JSON returned by Metaculus API to dataframe.

        :param questions_json: List of questions (as dicts)
        """
        questions_df = pd.DataFrame(questions_json)
        for col in ["created_time", "publish_time", "close_time", "resolve_time"]:
            questions_df[col] = questions_df[col].apply(pendulum.parse)

        questions_df["i_created"] = questions_df["author"] == self.user_id
        questions_df["i_predicted"] = questions_df["my_predictions"].apply(
            lambda x: x is not None
        )
        return questions_df
