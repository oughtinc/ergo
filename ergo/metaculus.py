import math
import functools
import json
import torch
import requests
import pendulum
import scipy
# scipy importing guidelines: https://docs.scipy.org/doc/scipy/reference/api.html
from scipy import stats
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

import pyro.distributions as dist
import ergo.logistic as logistic
import ergo.ppl as ppl

from typing import Optional, List, Any, Dict, Tuple

from typing_extensions import Literal
from dataclasses import dataclass, asdict


@dataclass
class ScoredPrediction:
    time: float
    prediction: Any
    resolution: float
    score: float
    question_name: str


class MetaculusQuestion:
    """
    Attributes:
    - url
    - page_url
    - id
    - author
    - title
    - status
    - resolution
    - created_time
    - publish_time
    - close_time
    - resolve_time
    - possibilities
    - can_use_powers
    - last_activity_time
    - activity
    - comment_count
    - votes
    - prediction_timeseries
    - author_name
    - prediction_histogram
    - anon_prediction_count
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
        r = self.metaculus.s.get(
            f"{self.metaculus.api_url}/questions/{self.id}")
        self.data = r.json()

    def sample_community(self):
        raise NotImplementedError("This should be implemented by a subclass")

    @staticmethod
    def to_dataframe(questions: List["MetaculusQuestion"]) -> pd.DataFrame:
        show_names = False
        for question in questions:
            if question.name:
                show_names = True
                break
        data = []
        if show_names:
            columns = ["id", "name", "title", "resolve_time"]
            for question in questions:
                data.append([question.id, question.name,
                             question.title, question.resolve_time])
        else:
            columns = ["id", "title", "resolve_time"]
            for question in questions:
                data.append(
                    [question.id, question.title, question.resolve_time])
        return pd.DataFrame(data, columns=columns)


class BinaryQuestion(MetaculusQuestion):
    def score_prediction(self, prediction, resolution) -> ScoredPrediction:
        predicted = prediction["x"]
        # Brier score, see https://en.wikipedia.org/wiki/Brier_score#Definition. 0 is best, 1 is worst, 0.25 is chance
        score = (resolution - predicted)**2
        return ScoredPrediction(prediction["t"], prediction, resolution, score, self.__str__())

    def get_scored_predictions(self):
        resolution = self.resolution
        if resolution is None:
            last_community_prediction = self.prediction_timeseries[-1]
            resolution = last_community_prediction["distribution"]["avg"]
        predictions = self.my_predictions["predictions"]
        return [self.score_prediction(prediction, resolution) for prediction in predictions]

    def submit(self, p: float):
        return self.metaculus.post(
            f"{self.metaculus.api_url}/questions/{self.id}/predict/",
            {
                "prediction": p,
                "void": False
            }
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

    # The Metaculus API accepts normalized predictions rather than predictions on the actual scale of the question
    # TODO: maybe it's better to fit the logistic first then normalize, rather than the other way around?
    def normalize_samples(self, samples):
        raise NotImplementedError("This should be implemented by a subclass")

    def get_submission_params(self, logistic_params: logistic.LogisticParams) -> SubmissionLogisticParams:
        distribution = stats.logistic(
            logistic_params.loc, logistic_params.scale)
        # The loc and scale have to be within a certain range for the Metaculus API to accept the prediction.

        # max loc of 3 set based on API response to prediction on https://pandemic.metaculus.com/questions/3920/what-will-the-cbo-estimate-to-be-the-cost-of-the-emergency-telework-act-s3561/
        clipped_loc = min(logistic_params.loc, 3)

        # max scale of 10 set based on API response to prediction on https://pandemic.metaculus.com/questions/3920/what-will-the-cbo-estimate-to-be-the-cost-of-the-emergency-telework-act-s3561/
        clipped_scale = min(max(logistic_params.scale, 0.01), 10)

        # We're not really sure what the deal with the low and high is.
        # Presumably they're supposed to be the points at which Metaculus "cuts off" your distribution
        # and ignores porbability mass assigned below/above.
        # But we're not actually trying to use them to "cut off" our distribution in a smart way;
        # we're just trying to include as much of our distribution as we can without the API getting unhappy
        # (we belive that if you set the low higher than the value below [or if you set the high lower],
        # then the API will reject the prediction, though we haven't tested that extensively)
        low = max(distribution.cdf(0), 0.01) if self.low_open else 0

        # min high of low + 0.01 set based on API response for https://www.metaculus.com/api2/questions/3961/predict/ --
        # {'prediction': ['high minus low must be at least 0.01']}"
        high = max(min(distribution.cdf(1), 0.99),
                   low + 0.01) if self.high_open else 1
        return SubmissionLogisticParams(clipped_loc, clipped_scale, low, high)

    # Map normalized samples back to actual scale
    def denormalize_samples(self, samples):
        raise NotImplementedError("This should be implemented by a subclass")

    @functools.lru_cache(None)
    def community_dist(self):
        # distribution on integers referencing 0...(len(self.prediction_histogram)-1)
        df = pd.DataFrame(self.prediction_histogram, columns=["x", "y1", "y2"])
        return dist.Categorical(probs=torch.Tensor(df["y2"]))

    def sample_community(self):
        normalized_sample = ppl.sample(
            self.community_dist()) / float(len(self.prediction_histogram))
        sample = self.denormalize_samples([normalized_sample])[0]
        if self.name:
            ppl.tag(sample, self.name)
        return sample

    def get_submission(self, mixture_params: logistic.LogisticMixtureParams) -> SubmissionMixtureParams:
        submission_logistic_params = [self.get_submission_params(
            logistic_params) for logistic_params in mixture_params.components]

        return SubmissionMixtureParams(submission_logistic_params, mixture_params.probs)

    def get_submission_from_samples(self, samples, samples_for_fit=5000) -> SubmissionMixtureParams:
        normalized_samples = self.normalize_samples(samples)
        mixture_params = logistic.fit_mixture(
            normalized_samples, num_samples=samples_for_fit)
        return self.get_submission(mixture_params)

    def show_submission(self, samples):
        raise NotImplementedError("This should be implemented by a subclass")

    @staticmethod
    def format_logistic_for_api(submission: SubmissionLogisticParams, weight: float) -> dict:

        # convert all the numbers to floats here so that you can be sure that wherever they originated
        # (e.g. numpy), they'll be regular old floats that can be converted to json by json.dumps
        return {
            "kind": "logistic",
            "x0": float(submission.loc),
            "s": float(submission.scale),
            "w": float(weight),
            "low": float(submission.low),
            "high": float(submission.high)
        }

    def submit(self, submission: SubmissionMixtureParams) -> requests.Response:
        prediction_data = {
            "prediction":
            {
                "kind": "multi",
                "d": [self.format_logistic_for_api(logistic_params, submission.probs[idx])
                      for idx, logistic_params in enumerate(submission.components)]
            },
            "void": False
        }

        r = self.metaculus.post(
            f"""{self.metaculus.api_url}/questions/{self.id}/predict/""",
            prediction_data
        )

        self.refresh_question()

        return r

    def submit_from_samples(self, samples, samples_for_fit=5000) -> requests.Response:
        submission = self.get_submission_from_samples(samples, samples_for_fit)
        return self.submit(submission)

    @staticmethod
    def get_logistic_from_json(logistic_json: Dict) -> SubmissionLogisticParams:
        return SubmissionLogisticParams(logistic_json["x0"], logistic_json["s"], logistic_json["low"], logistic_json["high"])

    @classmethod
    def get_submission_from_json(cls, submission_json: Dict) -> SubmissionMixtureParams:
        components = [cls.get_logistic_from_json(
            logistic_json) for logistic_json in submission_json]
        probs = [logistic_json["w"] for logistic_json in submission_json]
        return SubmissionMixtureParams(components, probs)

    def show_prediction(self, prediction: SubmissionMixtureParams):
        raise NotImplementedError("This should be implemented by a subclass")

    def get_latest_normalized_prediction(self) -> SubmissionMixtureParams:
        latest_prediction = self.my_predictions["predictions"][-1]["d"]
        return self.get_submission_from_json(latest_prediction)

    def show_community_prediction(self):
        raise NotImplementedError("This should be implemented by a subclass")

    # TODO: show vs. Metaculus and vs. resolution if available
    def show_performance(self):
        mixture_params = self.get_latest_normalized_prediction()
        self.show_prediction(mixture_params)


class LinearQuestion(ContinuousQuestion):
    def normalize_samples(self, samples):
        return (samples - self.question_range["min"]) / (self.question_range_width)

    def denormalize_samples(self, samples):
        return self.question_range["min"] + (self.question_range["max"] - self.question_range["min"]) * samples

    # Get the logistic on the actual scale of the question,
    # from the normalized logistic used in the submission
    # TODO: also return low and high on the true scale
    def get_true_scale_logistic_params(self, submission_logistic_params: SubmissionLogisticParams) -> logistic.LogisticParams:
        true_loc = submission_logistic_params.loc * \
            self.question_range_width + self.question_range["min"]

        true_scale = submission_logistic_params.scale * self.question_range_width

        return logistic.LogisticParams(true_loc, true_scale)

    def get_true_scale_mixture(self, submission_params: SubmissionMixtureParams) -> logistic.LogisticMixtureParams:
        true_scale_logistics_params = [self.get_true_scale_logistic_params(
            submission_logistic_params) for submission_logistic_params in submission_params.components]

        return logistic.LogisticMixtureParams(
            true_scale_logistics_params, submission_params.probs)

    def show_prediction(self, prediction: SubmissionMixtureParams, samples=None):
        true_scale_prediction = self.get_true_scale_mixture(prediction)

        pyplot.figure()
        pyplot.title(f"{self} prediction")  # type: ignore
        logistic.plot_mixture(true_scale_prediction,  # type: ignore
                              data=samples)  # type: ignore

    def show_submission(self, samples):
        submission = self.get_submission_from_samples(samples)

        self.show_prediction(submission, samples)

    def show_community_prediction(self, only_show_this=True):
        if only_show_this:
            pyplot.figure()
            pyplot.title(f"{self} community prediction")

        community_samples = [self.sample_community() for _ in range(0, 1000)]

        ax = seaborn.distplot(
            community_samples, label="Community")

        pyplot.legend()

        return ax


class LogQuestion(ContinuousQuestion):
    @property
    def deriv_ratio(self) -> float:
        return self.possibilities["scale"]["deriv_ratio"]

    def normalized_from_true_value(self, true_value) -> float:
        denominator = (
            true_value - self.question_range["min"])*(self.deriv_ratio - 1)
        timber = 1 + self.question_range_width/denominator
        return math.log(timber, self.deriv_ratio)

    def true_from_normalized_value(self, normalized_value):
        denominator = (self.deriv_ratio - 1) * \
            (self.deriv_ratio ** normalized_value - 1)
        scaled = self.question_range_width / denominator
        return self.question_range["min"] + scaled

    def normalize_samples(self, samples):
        return [self.normalized_from_true_value(sample) for sample in samples]

    def denormalize_samples(self, samples):
        return [self.true_from_normalized_value(sample) for sample in samples]

    def show_prediction(self, prediction: SubmissionMixtureParams, samples=None):
        prediction_samples = [logistic.sample_mixture(
            prediction) for _ in range(0, 5000)]

        true_scale_submission_samples = [self.true_from_normalized_value(
            submission_sample) for submission_sample in prediction_samples]

        pyplot.figure()
        pyplot.title(f"{self} prediction")  # type: ignore

        ax = seaborn.distplot(
            true_scale_submission_samples, label="Mixture")
        ax.set(xlabel='Sample value', ylabel='Density')
        ax.set_xlim(
            left=self.question_range["min"], right=self.question_range["max"])

        if samples is not None:
            seaborn.distplot(samples, label="Data")

        pyplot.xscale("log")  # type: ignore
        pyplot.legend()  # type: ignore

    def show_submission(self, samples):
        submission = self.get_submission_from_samples(samples)

        self.show_prediction(submission, samples)

    def show_community_prediction(self, only_show_this=True):
        if only_show_this:
            pyplot.figure()
            pyplot.title(f"{self} community prediction")

        community_samples = [self.sample_community() for _ in range(0, 1000)]
        pyplot.xscale("log")  # type: ignore

        ax = seaborn.distplot(
            community_samples, label="Community")

        pyplot.legend()

        return ax


class Metaculus:
    player_status_to_api_wording = {
        "predicted": "guessed_by",
        "not-predicted": "not_guessed_by",
        "author": "author",
        "interested": "upvoted_by",
    }

    def __init__(self, username, password, api_domain="www"):
        self.user_id = None
        self.api_url = f"https://{api_domain}.metaculus.com/api2"
        self.s = requests.Session()
        self.login(username, password)

    def login(self, username, password):
        loginURL = f"{self.api_url}/accounts/login/"
        r = self.s.post(loginURL,
                        headers={"Content-Type": "application/json", },
                        data=json.dumps({"username": username, "password": password}))

        self.user_id = r.json()['user_id']

    def post(self, url: str, data: Dict):
        r = self.s.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Referer": self.api_url,
                "X-CSRFToken": self.s.cookies.get_dict()["csrftoken"]
            },
            data=json.dumps(data)
        )
        try:
            r.raise_for_status()

        except requests.exceptions.HTTPError as e:
            e.args = (str(
                e.args), f"request body: {e.request.body}", f"response json: {e.response.json()}")
            raise

        return r

    def make_question_from_data(self, data=Dict, name=None) -> MetaculusQuestion:
        if(data["possibilities"]["type"] == "binary"):
            return BinaryQuestion(data["id"], self, data, name)
        if(data["possibilities"]["type"] == "continuous"):
            if(data["possibilities"]["scale"]["deriv_ratio"] != 1):
                return LogQuestion(data["id"], self, data, name)
            return LinearQuestion(data["id"], self, data, name)
        raise NotImplementedError(
            "We couldn't determine whether this question was binary, continuous, or something else")

    def get_question(self, id: int, name=None) -> MetaculusQuestion:
        r = self.s.get(f"{self.api_url}/questions/{id}")
        data = r.json()
        return self.make_question_from_data(data, name)

    def get_questions_json(self,
                           question_status: Literal["all", "upcoming", "open", "closed", "resolved", "discussion"] = "all",
                           player_status: Literal["any", "predicted",
                                                  "not-predicted", "author", "interested", "private"] = "any",
                           # 20 results per page
                           pages: int = 1,
                           ) -> List[Dict]:
        query_params = [f"status={question_status}", "order_by=-publish_time"]
        if player_status != "any":
            if player_status == "private":
                query_params.append("access=private")
            else:
                query_params.append(
                    f"{self.player_status_to_api_wording[player_status]}={self.user_id}")

        query_string = "&".join(query_params)

        def get_questions_for_pages(query_string: str, max_pages: int = 1, current_page: int = 1, results=[]) -> List[Dict]:
            if current_page > max_pages:
                return results

            r = self.s.get(
                f"{self.api_url}/questions/?{query_string}&page={current_page}")

            if r.json() == {"detail": "Invalid page."}:
                return results

            r.raise_for_status()

            return get_questions_for_pages(query_string, max_pages, current_page + 1, results + r.json()["results"])

        return get_questions_for_pages(query_string, pages)

    def make_questions_df(self, questions_json):
        questions_df = pd.DataFrame(questions_json)
        for col in ["created_time", "publish_time", "close_time", "resolve_time"]:
            questions_df[col] = questions_df[col].apply(pendulum.parse)

        questions_df["i_created"] = questions_df["author"] == self.user_id
        questions_df["i_predicted"] = questions_df["my_predictions"].apply(
            lambda x: x is not None)
        return questions_df
