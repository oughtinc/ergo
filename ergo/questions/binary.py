from abc import abstractmethod

import requests

from ergo.distributions.base import flip

from .question import Question


class BinaryQuestion(Question):
    @abstractmethod
    def get_community_prediction(self) -> float:
        """
        Get the latest community probability for the binary event
        """
        raise NotImplementedError("This should be implemented by a subclass")

    @abstractmethod
    def submit(self, p: float, confidence: float = 0) -> requests.Response:
        """
        Submit a prediction to the prediction platform

        :param p: how likely is the event to happen, from 0 to 1?
        """
        raise NotImplementedError("This should be implemented by a subclass")

    def sample_community(self) -> bool:
        """
        Sample from the PredictIt community distribution (Bernoulli).

        :return: true/false
        """
        return flip(self.get_community_prediction())
