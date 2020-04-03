import json
import requests
import pendulum
import scipy
# scipy importing guidelines: https://docs.scipy.org/doc/scipy/reference/api.html
from scipy import stats
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

import ergo.logistic as logistic

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

    def submit(self, p: float):
        return self.metaculus.post(
            f"{self.metaculus.api_url}/questions/{self.id}/predict/",
            {
                "prediction": p,
                "void": False
            }
        )


@dataclass
class ContinuousSubmission:
    loc: float
    scale: float
    low: float
    high: float


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

    # The Metaculus API accepts normalized predictions rather than predictions on the actual scale of the question
    def normalize_samples(self, samples, epsilon=1e-9):
        if self.is_log:
            samples = np.maximum(samples, epsilon)
            samples = samples / self.question_range["min"]
            return np.log(samples) / np.log(self.deriv_ratio)
        return (samples - self.question_range["min"]) / (self.question_range["max"] - self.question_range["min"])

    def get_loc_scale(self, samples) -> Tuple[float, float]:
        # TODO: Use logistic.fit_mixture, pass LogisticMixtureParams on to other functions
        normalized_samples = self.normalize_samples(samples)
        params = logistic.fit_single_scipy(normalized_samples)
        loc = params.loc
        scale = params.scale

        # The scale and loc have to be within a certain range for the Metaculus API to accept the prediction.
        # Based on playing with the API, we think that the ranges specified below are the widest possible.
        # TODO: confirm that this is actually true, could well be wrong
        clipped_loc = min(max(loc, -0.1565), 1.1565)
        clipped_scale = min(max(scale, 0.02), 10)
        return (clipped_loc, clipped_scale)

    def get_submission(self, scale, loc) -> ContinuousSubmission:
        distribution = stats.logistic(loc, scale)
        # We're not really sure what the deal with the low and high is.
        # Presumably they're supposed to be the points at which Metaculus "cuts off" your distribution
        # and ignores porbability mass assigned below/above.
        # But we're not actually trying to use them to "cut off" our distribution in a smart way;
        # we're just trying to include as much of our distribution as we can without the API getting unhappy
        # (we belive that if you set the low higher than the value below [or if you set the high lower],
        # then the API will reject the prediction, though we haven't tested that extensively)
        low = max(distribution.cdf(0), 0.01) if self.low_open else 0
        high = min(distribution.cdf(1), 0.99) if self.high_open else 1
        return ContinuousSubmission(loc, scale, low, high)

    def get_submission_from_samples(self, samples) -> ContinuousSubmission:
        loc, scale = self.get_loc_scale(samples)
        return self.get_submission(scale, loc)

    # Get the prediction on the actual scale of the question,
    # from the normalized prediction (Metaculus uses the normalized prediction)
    # TODO: instead of returning a regular logistic,
    # return a logistic that's cut off below the low and above the high, like for the Metaculus distribution
    def get_true_scale_prediction(self, normalized_s: float, normalized_x0: float):
        if self.is_log:
            raise NotImplementedError(
                "Scaling the normalized prediction to the true scale from the question not yet implemented for questions on the log scale")
        scaling_factor = self.question_range["max"] - \
            self.question_range["min"]

        def scale_param(param):
            return param * scaling_factor + self.question_range["min"]

        return stats.logistic(scale=scale_param(
            normalized_s), loc=scale_param(normalized_x0))

    def show_submission(self, samples):
        submission = self.get_submission_from_samples(samples)
        submission_rv = self.get_true_scale_prediction(
            submission.scale, submission.loc)
        pyplot.figure()
        pyplot.title(f"{self} prediction")
        seaborn.distplot(
            np.array(samples), label="samples")
        seaborn.distplot(np.array(submission_rv.rvs(1000)),
                         label="prediction")
        pyplot.legend()
        pyplot.show()

    def submit(self, submission: ContinuousSubmission) -> requests.Response:
        prediction_data = {
            "prediction":
            {
                "kind": "multi",
                "d": [
                    {
                        "kind": "logistic", "x0": submission.loc, "s": submission.scale, "w": 1, "low": submission.low, "high": submission.high
                    }
                ]
            },
            "void": False
        }

        return self.metaculus.post(
            f"""{self.metaculus.api_url}/questions/{self.id}/predict/""",
            prediction_data
        )

    def submit_from_samples(self, samples) -> requests.Response:
        submission = self.get_submission_from_samples(samples)
        return self.submit(submission)

    def score_prediction(self, prediction_dict: Dict, resolution: float) -> ScoredPrediction:
        # TODO: handle predictions with multiple distributions
        d = prediction_dict["d"][0]
        dist = stats.logistic(scale=d["s"], loc=d["x0"])
        score = dist.logpdf(resolution)
        return ScoredPrediction(prediction_dict["t"], prediction_dict, resolution, score, self.__str__())

    def get_scored_predictions(self):
        resolution = self.resolution
        if resolution is None:
            resolution = self.latest_community_prediction["q2"]
        predictions = self.my_predictions["predictions"]
        return [self.score_prediction(prediction, resolution) for prediction in predictions]

    # TODO: show vs. Metaculus and vs. resolution if available
    def show_performance(self):
        prediction = self.my_predictions["predictions"][0]
        d = prediction["d"][0]
        dist = self.get_true_scale_prediction(
            d["s"], d["x0"])
        pyplot.figure()
        pyplot.title(f"{self} latest prediction")
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
            return ContinuousQuestion(data["id"], self, data, name)
        raise NotImplementedError(
            "We couldn't determine whether this question was binary, continuous, or something else")

    def get_question(self, id: int, name=None) -> MetaculusQuestion:
        r = self.s.get(f"{self.api_url}/questions/{id}")
        data = r.json()
        return self.make_question_from_data(data, name)

    def get_prediction_results(self) -> pd.DataFrame:
        questions_data = self.get_questions_json(
            question_status="resolved", player_status="predicted", pages=9999)
        questions = [self.make_question_from_data(
            question_data) for question_data in questions_data]
        predictions_per_question = [
            question.get_scored_predictions() for question in questions]
        flat_predictions = [
            prediction for question in predictions_per_question for prediction in question]
        return pd.DataFrame([asdict(prediction) for prediction in flat_predictions])

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
