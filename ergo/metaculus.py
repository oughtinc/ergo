import json
import requests
import pendulum
import scipy
import pandas as pd
import numpy as np

from typing import Optional, List


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
        self.fetch()

    def api_url(self):
        return f"https://www.metaculus.com/api2/questions/{self.id}/"

    def fetch(self):
        self.data = requests.get(self.api_url()).json()
        return self.data

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
            print("Error on " + question.area)
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

    def show_submission(self, samples):
        pyplot.figure()
        submission = self.get_submission(samples)
        rv = scipy.stats.logistic(submission["loc"], submission["scale"])
        x = np.linspace(0, 1, 200)
        pyplot.plot(x, rv.pdf(x))
        sns.distplot(
            np.array(submission["normalized_samples"]), label="samples")
        sns.distplot(np.array(rv.rvs(1000)), label="prediction")
        pyplot.legend()
        pyplot.show()

    def submit(self, loc, scale):
        if not self.is_continuous:
            raise NotImplementedError("Can only submit continuous questions!")
        if not self.metaculus:
            raise ValueError(
                "Question was created without Metaculus connection!")

        scale = min(max(scale, 0.02), 10)
        loc = min(max(loc, 0), 1)
        distribution = scipy.stats.logistic(loc, scale)

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
        requests.post(
            f"""https://www.metaculus.com/api2/questions/{self.id}/predict/""",
            headers={
                "Content-Type": "application/json",
            },
            data=json.dumps(prediction_data)
        )

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
            loc, scale = scipy.stats.logistic.fit(samples)
            scale = min(max(scale, 0.02), 10)
            loc = min(max(loc, -0.1565), 1.1565)
            return loc, scale

    @classmethod
    def to_dataframe(self, questions: List["MetaculusQuestion"]):
        columns = ["id", "name", "title", "resolve_time"]
        data = []
        for question in questions:
            data.append([question.id, question.name,
                         question.title, question.resolve_time])
        return pd.DataFrame(data, columns=columns)


class BinaryQuestion(MetaculusQuestion):
    def score_prediction(self):
        resolution = self.data["resolution"]
        if resolution is None:
            last_community_prediction = self.data["prediction_timeseries"][-1]
            resolution = last_community_prediction["distribution"]["avg"]
        scores = []
        for my_prediction in self.data["my_predictions"]["predictions"]:
            value = my_prediction["x"]
            score = resolution * (1-value) ** 2 + (1-resolution) * (value) ** 2
            scores.append(score)
        return resolution, np.mean(scores)


class ContinuousQuestion(MetaculusQuestion):
    def score_prediction(self):
        resolution = self.data["resolution"]
        if resolution is None:
            last_community_prediction = self.data["prediction_timeseries"][-1]["community_prediction"]
            resolution = last_community_prediction["q2"]
        scores = []
        for my_prediction in self.data["my_predictions"]["predictions"]:
            # Todo: handle predictions with multiple distributions
            d = my_prediction["d"][0]
            dist = scipy.stats.logistic(scale=d["s"], loc=d["x0"])
            scores.append(dist.logpdf(resolution))
        return resolution, np.mean(scores)


class Metaculus:
    api_url = "https://www.metaculus.com/api2"

    def __init__(self, username, password):
        self.user_id = None
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
        # pp.pprint(data)
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

    def show_prediction_results(self):
        questions = self.get_predicted_questions()
        binary_questions = [
            question for question in questions if isinstance(question, BinaryQuestion)]
        continuous_questions = [
            question for question in questions if not isinstance(question, BinaryQuestion)]
        binary_scores = [question.score_prediction()
                         for question in binary_questions]
        continuous_scores = [question.score_prediction()
                             for question in continuous_questions]
        mean_binary_score = np.mean(binary_scores)
        mean_continuous_score = np.mean(continuous_scores)

        s = mean_binary_score
        p = (1 + np.sqrt(1-4*s))/2

        display(HTML(f"""<div>
        <strong>Summary</strong>
        <ol>
            <li>Mean Binary Score (lower is better): {mean_binary_score:.2}
            <ul>
                <li>Interpretation 1: we make calibrated forecasts of probability p={p:.2} (these predictions resolve true {p * 100:.0}% of the time)</li>
            </ul>
            </li>
            <li>Mean Continuous Score (higher is better): {mean_continuous_score:.2}</li>
        </ol>
        <br />
        </div>"""))

        for idx, question in enumerate(questions):
            resolution = question.data["resolution"]
            expected_resolution, score = question.score_prediction()

            display(HTML(f"""<div>
                <strong>{idx+1} - {question.data["title"]}</strong>
                <ol>
                    <li><em><a target="_blank" href="http://metaculus.com{question.data["page_url"]}">Metaculus link</a></em></li>
                    <li><em>open date</em>: {parse(question.data["publish_time"]).strftime("%b %d, %Y")}</li>
                    <li><em>resolve date</em>: {parse(question.data["resolve_time"]).strftime("%b %d, %Y")}</li>
                    <li><em>type</em>: {question.data["possibilities"]["type"]}</li>
                    <li><em>resolution</em>: {resolution}</li>"""))

            if isinstance(question, BinaryQuestion):
                if resolution is None:
                    display(HTML(
                        '<li><em>community expected resolution</em>: {:.2f}</li>'.format(expected_resolution)))
                    display(
                        HTML('<li><em>brier score (lower is better)</em>: {:.3f}</li>'.format(score)))
            else:
                if resolution is None:
                    display(HTML(
                        '<li><em>community expected resolution</em>: {:.2f}</li>'.format(expected_resolution)))
                    display(
                        HTML('<li><em>log pdf score (higher is better)</em>: {:.3f}</li>'.format(score)))
                display(HTML(f"""</ol>
                <br />"""))
