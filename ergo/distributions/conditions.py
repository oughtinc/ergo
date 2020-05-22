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

    @abstractmethod
    def get_normalized(self, scale_min: float, scale_max: float):
        """
        Assume that the condition's true range is [scale_min, scale_max].
        Return the normalized condition.

        :param scale_min: the true-scale minimum of the range
        :param scale_max: the true-scale maximum of the range
        :return: the condition normalized to [0,1]
        """

    @abstractmethod
    def get_denormalized(self, scale_min: float, scale_max: float):
        """
        Assume that the condition has been normalized to be over [0,1].
        Return the condition on the true scale.

        :param scale_min: the true-scale minimum of the range
        :param scale_max: the true-scale maximum of the range
        :return: the condition on the true scale of [scale_min, scale_max]
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

    def get_normalized(self, scale_min: float, scale_max: float):
        scale_range = scale_max - scale_min
        normalized_min = (
            ((self.min - scale_min) / scale_range) if self.min is not None else None
        )
        normalized_max = (
            ((self.max - scale_min) / scale_range) if self.max is not None else None
        )
        return self.__class__(self.p, normalized_min, normalized_max, self.weight)

    def get_denormalized(self, scale_min: float, scale_max: float):
        scale_range = scale_max - scale_min
        denormalized_min = (
            (self.min * scale_range) + scale_min if self.min is not None else None
        )
        denormalized_max = (
            (self.max * scale_range) + scale_min if self.max is not None else None
        )
        return self.__class__(self.p, denormalized_min, denormalized_max, self.weight)

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

    def get_normalized(self, scale_min: float, scale_max: float):
        scale_range = scale_max - scale_min
        normalized_histogram: Histogram = [
            {
                "x": (entry["x"] - scale_min) / scale_range,
                "density": entry["density"] * scale_range,
            }
            for entry in self.histogram
        ]
        return self.__class__(normalized_histogram, self.weight)

    def get_denormalized(self, scale_min: float, scale_max: float):
        scale_range = scale_max - scale_min
        denormalized_histogram: Histogram = [
            {
                "x": (entry["x"] * scale_range) + scale_min,
                "density": entry["density"] / scale_range,
            }
            for entry in self.histogram
        ]
        return self.__class__(denormalized_histogram, self.weight)

    def __str__(self):
        return "The probability density function looks similar to the provided density function."


@dataclass
class ScalePriorCondition(Condition):
    """
    Condition that enforces prior on scales for mixture distributions
    """

    weight: float = 1.0
    scale_mean: float = 1.0

    def loss(self, dist):
        total_loss = 0.0
        for component in dist.components:
            scale_penalty = (self.scale_mean - component.scale) ** 2
            total_loss += scale_penalty
        return self.weight * total_loss

    def get_normalized(self, scale_min: float, scale_max: float):
        scale_range = scale_max - scale_min
        return self.__class__(self.weight, self.scale_mean / scale_range)

    def get_denormalized(self, scale_min, scale_max):
        scale_range = scale_max - scale_min
        return self.__class__(self.weight, self.scale_mean * scale_range)

    def __str__(self):
        return f"The scale is normally distributed around {self.scale_mean}"


@dataclass
class LocationPriorCondition(Condition):
    """
    Condition that enforces prior on scales for mixture distributions
    """

    weight: float = 1.0
    loc_mean: float = 0.0

    def loss(self, dist):
        total_loss = 0.0
        for component in dist.components:
            loc_penalty = (self.loc_mean - component.loc) ** 2
            total_loss += loc_penalty
        return self.weight * total_loss

    def get_normalized(self, scale_min: float, scale_max: float):
        scale_range = scale_max - scale_min
        return self.__class__(self.weight, (self.loc_mean - scale_min) / scale_range)

    def get_denormalized(self, scale_min, scale_max):
        scale_range = scale_max - scale_min
        return self.__class__(self.weight, (self.loc_mean * scale_range) + scale_min)

    def __str__(self):
        return f"The location is normally distributed around {self.loc_mean}"
