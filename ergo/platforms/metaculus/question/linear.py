from dataclasses import dataclass
from typing import Any, Dict

from ergo.distributions import Logistic, LogisticMixture
from ergo.scale import Scale

from .continuous import ContinuousQuestion


@dataclass
class LinearQuestion(ContinuousQuestion):
    """
    A continuous Metaculus question that's on a linear (as opposed to a log) scale"
    """

    scale: Scale

    def __init__(
        self, id: int, metaculus: Any, data: Dict, name=None,
    ):
        super().__init__(id, metaculus, data, name)
        self.scale = Scale(
            float(self.question_range["min"]), float(self.question_range["max"])
        )

    # TODO: also return low and high on the true scale,
    # and use those somehow in logistic.py
    def get_true_scale_logistic(self, normalized_dist: Logistic) -> Logistic:
        """
        Convert a normalized logistic distribution to a logistic on
        the true scale of the question.

        :param normalized_dist: normalized logistic distribution
        :return: logistic distribution on the true scale of the question
        """

        return normalized_dist.denormalize(self.scale)

    def get_true_scale_mixture(
        self, normalized_dist: LogisticMixture
    ) -> LogisticMixture:
        """
        Convert a normalized logistic mixture distribution to a
        logistic on the true scale of the question.

        :param normalized_dist: normalized logistic mixture dist
        :return: same distribution rescaled to the true scale of the question
        """

        return normalized_dist.denormalize(self.scale)
