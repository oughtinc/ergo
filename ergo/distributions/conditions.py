from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, TypeVar

from jax import jit, vmap
import jax.numpy as np
import numpy as onp

from ergo.utils import shift

from . import histogram
from .scale import Scale

ConditionClass = TypeVar("ConditionClass", bound="Condition")
ScaleClass = TypeVar("ScaleClass", bound=Scale)


def static_value(v):
    if isinstance(v, np.DeviceArray) or isinstance(v, onp.ndarray):
        return tuple(v)
    else:
        return v


class Condition(ABC):
    weight: float = 1.0

    def __init__(self, weight=1.0):
        self.weight = weight

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Condition):
            return self.__key() == other.__key()
        return NotImplemented

    def __key(self):
        cls, params = self.destructure()
        return (cls, tuple(static_value(param) for param in params))

    def _describe_fit(self, dist) -> Dict[str, Any]:
        # convert to float for easy serialization
        return {"loss": self.loss(dist)}

    def normalize(self, scale: ScaleClass):
        """
        Assume that the condition's true range is [scale_min, scale_max].
        Return the normalized condition.

        :param scale: the true-scale
        :return: the condition normalized to [0,1]
        """
        return self

    def denormalize(self, scale: ScaleClass):
        """
        Assume that the condition has been normalized to be over [0,1].
        Return the condition on the true scale.

        :param scale: the true-scale
        :return: the condition on the true scale of [scale_min, scale_max]
        """
        return self

    def describe_fit(self, dist) -> Dict[str, float]:
        """
        Describe how well the distribution meets the condition

        :param dist: A probability distribution
        :return: A description of various aspects of how well
        the distribution meets the condition
        """

        result = static_describe_fit(*dist.destructure(), *self.destructure())
        return {k: float(v) for (k, v) in result.items()}

    @abstractmethod
    def loss(self, dist):
        """
        Loss function for this condition when fitting a distribution.

        Should have max loss = 1 without considering weight
        Should multiply loss * weight

        :param dist: A probability distribution
        """

    def shape_key(self):
        return (self.__class__.__name__,)

    @abstractmethod
    def destructure(self) -> Tuple[ConditionClass, Sequence[Any]]:
        ...

    @classmethod
    def structure(cls, params) -> "Condition":
        return cls(*params)


class SmoothnessCondition(Condition):
    def loss(self, dist) -> float:
        window_size = 5
        squared_distance = 0.0
        for i in range(1, window_size + 1):
            squared_distance += (1 / i ** 2) * np.sum(
                np.square(dist.logps - shift(dist.logps, i, dist.logps[0]))
            )
        return self.weight * squared_distance / dist.logps.size

    def destructure(self):
        return (SmoothnessCondition, (self.weight,))

    def __str__(self):
        return "Minimize rough edges in the distribution"


class MaxEntropyCondition(Condition):
    def loss(self, dist) -> float:
        return -self.weight * dist.entropy()

    def destructure(self):
        return (MaxEntropyCondition, (self.weight,))

    def __str__(self):
        return "Maximize the entropy of the distribution"


class CrossEntropyCondition(Condition):
    p_dist: "histogram.HistogramDist"
    weight: float = 1.0

    def __init__(self, p_dist, weight=1.0):
        self.p_dist = p_dist
        super().__init__(weight)

    def loss(self, q_dist) -> float:
        return self.weight * self.p_dist.cross_entropy(q_dist)

    def destructure(self):
        return (CrossEntropyCondition, (np.array(self.p_dist.logps), self.weight))

    @classmethod
    def structure(cls, params):
        return cls(histogram.HistogramDist(params[0], traceable=True), params[1])

    def __str__(self):
        return "Minimize the cross-entropy of the two distributions"


@jit
def wasserstein_distance(xs, ys):
    diffs = np.cumsum(xs - ys)
    abs_diffs = np.abs(diffs)
    return np.sum(abs_diffs)


class WassersteinCondition(Condition):
    p_dist: "histogram.HistogramDist"
    weight: float = 1.0

    def __init__(self, p_dist, weight=1.0):
        self.p_dist = p_dist
        super().__init__(weight)

    def loss(self, q_dist) -> float:
        return self.weight * wasserstein_distance(self.p_dist.ps, q_dist.ps)

    def destructure(self):
        return (WassersteinCondition, (np.array(self.p_dist.logprobs), self.weight))

    @classmethod
    def structure(cls, params):
        return cls(histogram.HistogramDist(params[0]), params[1], traceable=True)

    def __str__(self):
        return "Minimize the Wasserstein distance between the two distributions"


class IntervalCondition(Condition):
    """
    The specified interval should include as close to the specified
    probability mass as possible

    :raises ValueError: max must be strictly greater than min
    """

    p: float
    min: Optional[float]
    max: Optional[float]
    weight: float

    def __init__(self, p, min=None, max=None, weight=1.0):
        self.p = p
        self.min = min
        self.max = max
        super().__init__(weight)

    def actual_p(self, dist) -> float:
        cdf_at_min = dist.cdf(self.min) if self.min is not None else 0
        cdf_at_max = dist.cdf(self.max) if self.max is not None else 1
        return cdf_at_max - cdf_at_min

    def loss(self, dist):
        actual_p = self.actual_p(dist)
        return self.weight * (actual_p - self.p) ** 2

    def _describe_fit(self, dist):
        description = super()._describe_fit(dist)
        description["p_in_interval"] = self.actual_p(dist)
        return description

    def normalize(self, scale: ScaleClass):
        normalized_min = scale.normalize_point(self.min)
        normalized_max = scale.normalize_point(self.max)
        return self.__class__(self.p, normalized_min, normalized_max, self.weight)

    def denormalize(self, scale: ScaleClass):
        denormalized_min = scale.denormalize_point(self.min)
        denormalized_max = scale.denormalize_point(self.max)
        return self.__class__(self.p, denormalized_min, denormalized_max, self.weight)

    def destructure(self):
        return (IntervalCondition, (self.p, self.min, self.max, self.weight))

    def shape_key(self):
        return (self.__class__.__name__, self.min is None, self.max is None)

    def __str__(self):
        return f"There is a {self.p:.0%} chance that the value is in [{self.min}, {self.max}]"


class HistogramCondition(Condition):
    """
    The distribution should fit the specified histogram as closely as
    possible
    """

    xs: np.DeviceArray
    densities: np.DeviceArray
    weight: float = 1.0

    def __init__(self, xs, densities, weight=1.0):
        self.xs = xs
        self.densities = densities
        super().__init__(weight)

    def loss(self, dist):
        entry_loss_fn = lambda x, density: (density - dist.pdf1(x)) ** 2  # noqa: E731
        total_loss = np.sum(vmap(entry_loss_fn)(self.xs, self.densities))
        return self.weight * total_loss / self.xs.size

    def normalize(self, scale: ScaleClass):
        normalized_xs = np.array([scale.normalize_point(x) for x in self.xs])
        normalized_densities = np.array(
            [density * scale.scale_range for density in self.densities]
        )
        return self.__class__(normalized_xs, normalized_densities, self.weight)

    def denormalize(self, scale: ScaleClass):
        denormalized_xs = np.array([scale.denormalize_point(x) for x in self.xs])
        denormalized_densities = np.array(
            [density / scale.scale_range for density in self.densities]
        )
        return self.__class__(denormalized_xs, denormalized_densities, self.weight)

    def destructure(self):
        return (HistogramCondition, (self.xs, self.densities, self.weight))

    def __key(self):
        return (
            HistogramCondition,
            (tuple(self.xs), tuple(self.densities), self.weight),
        )

    def __str__(self):
        return "The probability density function looks similar to the provided density function."


@partial(jit, static_argnums=(0, 2, 3))
def static_describe_fit(dist_class, dist_params, scale, cond_class, cond_params):
    dist = dist_class.structure(*dist_params, scale)
    condition = cond_class.structure(cond_params)
    return condition._describe_fit(dist)
