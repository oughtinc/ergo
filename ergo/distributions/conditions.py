from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from jax import vmap
import jax.numpy as np

from .types import Histogram


class Condition(ABC):
    @abstractmethod
    def loss(self, dist):
        """
        Loss function for this condition when fitting a distribution.

        Should have max loss = 1 without considering weight
        Should multiply loss * weight

        :param dist: A probability distribution
        """

    def describe_fit(self, dist) -> Dict[str, Any]:
        """
        Describe how well the distribution meets the condition

        :param dist: A probability distribution
        :return: A description of various aspects of how well
        the distribution meets the condition
        """

        # convert to float for easy serialization
        return {"loss": float(self.loss(dist))}


@dataclass
class IntervalCondition(Condition):
    """
    Condition that the specified interval should include
    as close to the specified probability mass as possible

    :raises ValueError: max must be strictly greater than min
    """

    p: float
    min: Optional[float]
    max: Optional[float]
    weight: float

    def __init__(self, p, min=None, max=None, weight=1.0):
        if min is not None and max is not None and max <= min:
            raise ValueError(
                f"max must be strictly greater than min, got max: {max}, min: {min}"
            )

        self.p = p
        self.min = min
        self.max = max
        self.weight = weight

    def actual_p(self, dist) -> float:
        cdf_at_min = dist.cdf(self.min) if self.min is not None else 0
        cdf_at_max = dist.cdf(self.max) if self.max is not None else 1
        return cdf_at_max - cdf_at_min

    def loss(self, dist):
        actual_p = self.actual_p(dist)
        return self.weight * (actual_p - self.p) ** 2

    def describe_fit(self, dist):
        description = super().describe_fit(dist)

        # report the actual probability mass in the interval
        description["p_in_interval"] = float(self.actual_p(dist))
        return description

    def __str__(self):
        return f"There is a {self.p:.0%} chance that the value is in [{self.min}, {self.max}]"


@dataclass
class HistogramCondition(Condition):
    """
    Condition that the distribution should fit the specified histogram
    as closely as possible
    """

    histogram: Histogram
    weight: float = 1.0

    def _loss_loop(self, dist):
        total_loss = 0.0
        for entry in self.histogram:
            target_density = entry["density"]
            actual_density = dist.pdf1(entry["x"])
            total_loss += (actual_density - target_density) ** 2
        return self.weight * total_loss / len(self.histogram)

    def loss(self, dist):
        xs = np.array([entry["x"] for entry in self.histogram])
        densities = np.array([entry["density"] for entry in self.histogram])
        entry_loss_fn = lambda x, density: (density - dist.pdf1(x)) ** 2  # noqa: E731
        total_loss = np.sum(vmap(entry_loss_fn)(xs, densities))
        return self.weight * total_loss / len(self.histogram)

    def __str__(self):
        return "The probability density function looks similar to the provided density function."
