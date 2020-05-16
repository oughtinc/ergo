from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

from ergo.distributions.base import flip

from .question import MetaculusQuestion


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
