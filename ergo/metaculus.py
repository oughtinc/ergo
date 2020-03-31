import json
import requests
import pendulum
import scipy
# scipy importing guidelines: https://docs.scipy.org/doc/scipy/reference/api.html
from scipy import stats
import seaborn  # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

from typing import Optional, List, Any
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
    def min(self) -> Optional[float]:
        if self.is_continuous:
            return self.possibilities["scale"]["min"]
        return None

    @property
    def max(self) -> Optional[float]:
        if self.is_continuous:
            return self.possibilities["scale"]["max"]
        return None

    @property
    def deriv_ratio(self) -> Optional[float]:
        if self.is_continuous:
            return self.possibilities["scale"]["deriv_ratio"]
        return None

    @property
    def is_log(self) -> bool:
        if self.is_continuous:
            return self.deriv_ratio != 1
        return False

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

    def get_scored_predictions(self) -> List[ScoredPrediction]:
        raise NotImplementedError("This should be implemented by a subclass")

    @staticmethod
    def to_dataframe(questions: List["MetaculusQuestion"]) -> pd.DataFrame:
        columns = ["id", "name", "title", "resolve_time"]
        data = []
        for question in questions:
            data.append([question.id, question.name,
                         question.title, question.resolve_time])
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


class ContinuousQuestion(MetaculusQuestion):
    @property
    def low_open(self) -> bool:
        return self.possibilities["low"] == "tail"

    @property
    def high_open(self) -> bool:
        return self.possibilities["high"] == "tail"

    @property
    def metaculus_scale(self):
        return self.possibilities["scale"]

    def show_raw_prediction(self, samples):
        pyplot.figure()
        pyplot.title(str(self))
        loc, scale = stats.logistic.fit(samples)
        prediction_rv = stats.logistic(loc, scale)
        seaborn.distplot(
            samples, label="samples")
        seaborn.distplot(np.array(prediction_rv.rvs(1000)), label="prediction")
        # community_dist = self.hydrate_prediction(
        #     self.latest_community_prediction)
        # seaborn.distplot(np.array(community_dist.rvs(1000)),
        #                  label="community prediction")
        pyplot.legend()
        pyplot.show()

    def normalize_samples(self, samples, epsilon=1e-9):
        if self.is_continuous:
            if self.is_log:
                samples = np.maximum(samples, epsilon)
                samples = samples / self.min
                samples = np.log(samples) / np.log(self.deriv_ratio)
            else:
                samples = (samples - self.min) / (self.max - self.min)
        return samples

    def fit_single_logistic(self, samples):
        with np.errstate(all='raise'):
            loc, scale = stats.logistic.fit(samples)
            scale = min(max(scale, 0.02), 10)
            loc = min(max(loc, -0.1565), 1.1565)
            return loc, scale

    def get_prediction(self, samples):
        try:
            normalized_samples = self.normalize_samples(samples)
            loc, scale = self.fit_single_logistic(normalized_samples)
        except FloatingPointError:
            print("Error on " + self.area)
            traceback.print_exc()
        else:
            return {
                "normalized_samples": normalized_samples,
                "loc": loc,
                "scale": scale
            }

    def show_scaled_submission(self, samples):
        pyplot.figure()
        submission = self.get_prediction(samples)
        submission_rv = stats.logistic(submission["loc"], submission["scale"])
        seaborn.distplot(
            np.array(submission["normalized_samples"]), label="samples")
        seaborn.distplot(np.array(submission_rv.rvs(1000)), label="prediction")
        pyplot.legend()
        pyplot.show()

    def submit(self, loc, scale):
        submission = self.get_submission(loc, scale)
        prediction_data = {
            "prediction":
            {
                "kind": "multi",
                "d": [
                    {
                        "kind": "logistic", "x0": submission["submission_loc"], "s": submission["submission_scale"], "w": 1, "low": submission["low"], "high": submission["high"]
                    }
                ]
            },
            "void": False
        }

        r = self.metaculus.s.post(
            f"""{self.metaculus.api_url}/questions/{self.id}/predict/""",
            headers={
                "Content-Type": "application/json",
                "Referer": self.metaculus.api_url,
                "X-CSRFToken": self.metaculus.s.cookies.get_dict()["csrftoken"]
            },
            data=json.dumps(prediction_data)
        )
        try:
            r.raise_for_status()

        except requests.exceptions.HTTPError as e:
            e.args = (str(
                e.args), f"request body: {e.request.body}", f"response json: {e.response.json()}")
            raise

        return r

    def get_submission(self, submission_loc, submission_scale):
        if not self.is_continuous:
            raise NotImplementedError("Can only submit continuous questions!")

        submission_scale = min(max(submission_scale, 0.02), 10)
        submission_loc = min(max(submission_loc, 0), 1)
        distribution = stats.logistic(submission_loc, submission_scale)

        low = max(distribution.cdf(0), 0.01) if self.low_open else 0
        high = min(distribution.cdf(1), 0.99) if self.high_open else 1
        return {
            "submission_scale": submission_scale,
            "submission_loc": submission_loc,
            "low": low,
            "high": high
        }

    def submit_from_samples(self, samples):
        submission = self.get_prediction(samples)
        return self.submit(submission["loc"], submission["scale"])

    def score_prediction(self, prediction, resolution) -> ScoredPrediction:
        # TODO: handle predictions with multiple distributions
        d = prediction["d"][0]
        dist = stats.logistic(scale=d["s"], loc=d["x0"])
        score = dist.logpdf(resolution)
        return ScoredPrediction(prediction["t"], prediction, resolution, score, self.__str__())

    def get_scored_predictions(self):
        resolution = self.resolution
        if resolution is None:
            resolution = self.latest_community_prediction["q2"]
        predictions = self.my_predictions["predictions"]
        return [self.score_prediction(prediction, resolution) for prediction in predictions]

    def hydrate_prediction(self, prediction):
        scaling_factor = self.metaculus_scale["max"] - \
            self.metaculus_scale["min"]

        def scale_param(param):
            return param * scaling_factor + self.metaculus_scale["min"]

        d = prediction["d"][0]
        return stats.logistic(scale=scale_param(
            d["s"]), loc=scale_param(d["x0"]))

    def show_performance(self):
        prediction = self.my_predictions["predictions"][0]
        dist = self.hydrate_prediction(prediction)
        pyplot.figure()
        seaborn.distplot(np.array(dist.rvs(1000)), label="prediction")
        pyplot.legend()
        pyplot.show()


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

    def get_question(self, id: int, name=None) -> MetaculusQuestion:
        r = self.s.get(f"{self.api_url}/questions/{id}")
        data = r.json()
        if(data["possibilities"]["type"] == "binary"):
            return BinaryQuestion(id, self, data, name)
        if(data["possibilities"]["type"] == "continuous"):
            return ContinuousQuestion(id, self, data, name)
        raise NotImplementedError(
            "We couldn't determine whether this question was binary, continuous, or something else")

    def get_predicted_questions(self) -> List[MetaculusQuestion]:
        r = self.s.get(f"{self.api_url}/questions/?guessed_by={self.user_id}")
        json_data = r.json()
        results = json_data["results"]
        question_ids = [result["id"] for result in results]
        return [self.get_question(id) for id in question_ids]
        # TODO: retrieve additional data if next link is not None

    def get_prediction_results(self) -> pd.DataFrame:
        questions = self.get_predicted_questions()
        predictions_per_question = [
            question.get_scored_predictions() for question in questions]
        flat_predictions = [
            prediction for question in predictions_per_question for prediction in question]
        return pd.DataFrame([asdict(prediction) for prediction in flat_predictions])

    def get_questions_for_pages(self, query_string: str, max_pages: int = 1, current_page: int = 1, results=[]) -> List[MetaculusQuestion]:
        if current_page > max_pages:
            return results

        r = self.s.get(
            f"{self.api_url}/questions/?{query_string}&page={current_page}")

        if r.json() == {"detail": "Invalid page."}:
            return results

        r.raise_for_status()

        return self.get_questions_for_pages(query_string, max_pages, current_page + 1, results + r.json()["results"])

    def get_questions(self,
                      question_status: Literal["all", "upcoming", "open", "closed", "resolved", "discussion"] = "all",
                      player_status: Literal["any", "predicted",
                                             "not-predicted", "author", "interested", "private"] = "any",
                      # 20 results per page
                      pages: int = 1
                      ) -> pd.DataFrame:
        query_params = [f"status={question_status}", "order_by=-publish_time"]
        if player_status != "any":
            if player_status == "private":
                query_params.append("access=private")
            else:
                query_params.append(
                    f"{self.player_status_to_api_wording[player_status]}={self.user_id}")

        query_string = "&".join(query_params)
        questions_json = self.get_questions_for_pages(query_string, pages)
        questions_df = pd.DataFrame(questions_json)
        for col in ["created_time", "publish_time", "close_time", "resolve_time"]:
            questions_df[col] = questions_df[col].apply(pendulum.parse)

        questions_df["i_created"] = questions_df["author"] == self.user_id
        questions_df["i_predicted"] = questions_df["my_predictions"].apply(
            lambda x: x is not None)
        return questions_df
