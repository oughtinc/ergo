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
    def is_continuous(self):
        return self.type == "continuous"

    @property
    def type(self):
        return self.possibilities["type"]

    @property
    def min(self):
        if self.is_continuous:
            return self.possibilities["scale"]["min"]
        return None

    @property
    def max(self):
        if self.is_continuous:
            return self.possibilities["scale"]["max"]
        return None

    @property
    def deriv_ratio(self):
        if self.is_continuous:
            return self.possibilities["scale"]["deriv_ratio"]
        return None

    @property
    def is_log(self):
        if self.is_continuous:
            return self.deriv_ratio != 1
        return False

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

    def get_submission(self, samples):
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

    def submit_from_samples(self, samples):
        submission = self.get_submission(samples)
        self.submit(submission["loc"], submission["scale"])
        return (submission["loc"], submission["scale"])

    def show_submission(self, samples):
        pyplot.figure()
        submission = self.get_submission(samples)
        rv = stats.logistic(submission["loc"], submission["scale"])
        x = np.linspace(0, 1, 200)
        pyplot.plot(x, rv.pdf(x))
        seaborn.distplot(
            np.array(submission["normalized_samples"]), label="samples")
        seaborn.distplot(np.array(rv.rvs(1000)), label="prediction")
        pyplot.legend()
        pyplot.show()

    def submit(self, loc, scale):
        if not self.is_continuous:
            raise NotImplementedError("Can only submit continuous questions!")

        scale = min(max(scale, 0.02), 10)
        loc = min(max(loc, 0), 1)
        distribution = stats.logistic(loc, scale)

        low = max(distribution.cdf(0), 0.01)
        high = min(distribution.cdf(1), 0.99)
        prediction_data = {
            "prediction":
            {
                "kind": "multi",
                "d": [
                    {
                        "kind": "logistic", "x0": loc, "s": scale, "w": 1, "low": low, "high": high
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

        except:
            print(r.json())
            raise

        return r

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

    def get_scored_predictions(self):
        pass

    @staticmethod
    def to_dataframe(questions: List["MetaculusQuestion"]):
        columns = ["id", "name", "title", "resolve_time"]
        data = []
        for question in questions:
            data.append([question.id, question.name,
                         question.title, question.resolve_time])
        return pd.DataFrame(data, columns=columns)


@dataclass
class ScoredPrediction:
    time: float
    prediction: Any
    resolution: float
    score: float
    question_name: str


class BinaryQuestion(MetaculusQuestion):
    def score_prediction(self, prediction, resolution) -> ScoredPrediction:
        predicted = prediction["x"]
        # Brier score, see https://en.wikipedia.org/wiki/Brier_score#Definition. 0 is best, 1 is worst, 0.25 is chance
        score = (resolution - predicted)**2
        return ScoredPrediction(prediction["t"], prediction, resolution, score, self.__str__())

    def get_scored_predictions(self):
        # print(self.data)
        resolution = self.data["resolution"]
        if resolution is None:
            last_community_prediction = self.data["prediction_timeseries"][-1]
            resolution = last_community_prediction["distribution"]["avg"]
        predictions = self.data["my_predictions"]["predictions"]
        return [self.score_prediction(prediction, resolution) for prediction in predictions]


class ContinuousQuestion(MetaculusQuestion):
    def score_prediction(self, prediction, resolution) -> ScoredPrediction:
        # TODO: handle predictions with multiple distributions
        d = prediction["d"][0]
        dist = stats.logistic(scale=d["s"], loc=d["x0"])
        score = dist.logpdf(resolution)
        return ScoredPrediction(prediction["t"], prediction, resolution, score, self.__str__())

    def get_scored_predictions(self):
        resolution = self.data["resolution"]
        if resolution is None:
            last_community_prediction = self.data["prediction_timeseries"][-1]["community_prediction"]
            resolution = last_community_prediction["q2"]
        predictions = self.data["my_predictions"]["predictions"]
        return [self.score_prediction(prediction, resolution) for prediction in predictions]


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

    def get_question(self, id: int, name=None):
        r = self.s.get(f"{self.api_url}/questions/{id}")
        data = r.json()
        if(data["possibilities"]["type"] == "binary"):
            return BinaryQuestion(id, self, data, name)
        if(data["possibilities"]["type"] == "continuous"):
            return ContinuousQuestion(id, self, data, name)
        raise NotImplementedError(
            "We couldn't determine whether this question was binary, continuous, or something else")

    def get_predicted_questions(self):
        r = self.s.get(f"{self.api_url}/questions/?guessed_by={self.user_id}")
        json_data = r.json()
        results = json_data["results"]
        question_ids = [result["id"] for result in results]
        return [self.get_question(id) for id in question_ids]
        # TODO: retrieve additional data if next link is not None

    def get_prediction_results(self):
        questions = self.get_predicted_questions()
        predictions_per_question = [
            question.get_scored_predictions() for question in questions]
        flat_predictions = [
            prediction for question in predictions_per_question for prediction in question]
        return pd.DataFrame([asdict(prediction) for prediction in flat_predictions])

    def get_questions_for_pages(self, query_string: str, max_pages: int = 1, current_page: int = 1, results=[]):
        if current_page > max_pages:
            return results

        r = self.s.get(
            f"{self.api_url}/questions/?{query_string}&page={current_page}")
        if r.status_code < 400:
            return self.get_questions_for_pages(query_string, max_pages, current_page + 1, results + r.json()["results"])

        if r.json() == {"detail": "Invalid page."}:
            return results

        r.raise_for_status()

    def get_questions(self,
                      question_status: Literal["all", "upcoming", "open", "closed", "resolved", "discussion"] = "all",
                      player_status: Literal["any", "predicted",
                                             "not-predicted", "author", "interested", "private"] = "any",
                      # 20 results per page
                      pages: int = 1
                      ):
        query_params = [f"status={question_status}", "order_by=-publish_time"]
        if player_status != "any":
            if player_status == "private":
                query_params.append("access=private")
            else:
                query_params.append(
                    f"{self.player_status_to_api_wording[player_status]}={self.user_id}")

        query_string = "&".join(query_params)
        return self.get_questions_for_pages(query_string, pages)
