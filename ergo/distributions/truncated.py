from dataclasses import dataclass
from .distribution import Distribution
import jax.numpy as np
from .conditions import Condition
from typing import Sequence, Optional


class TruncatedDist(Distribution):
    underlying_dist: Distribution
    floor: float
    ceiling: float

    def ppf(self, q):
        raise NotImplementedError

    def pdf1(self, x):
        p_below = self.underlying_dist.cdf(self.floor)
        p_above = 1 - self.underlying_dist.cdf(self.ceiling)
        p_inside = 1 - (p_below + p_above)

        p_at_x = self.underlying_dist.pdf1(x) * (1 / p_inside)

        return np.where(x < self.floor, 0, np.where(x > self.ceiling, 0, p_at_x))

    @classmethod
    def from_params(cls, params):
        underlying_dist = params.underlying_dist.from_params(params)
        return cls(underlying_dist, **params)

    @classmethod
    def from_conditions(cls, underlying_dist_class, floor, ceiling, **kwargs):
        underlying_dist = underlying_dist_class.from_conditions(  # type: ignore
            containing_class=cls, scale_min=floor, scale_max=ceiling, **kwargs
        )
        return cls(underlying_dist, floor, ceiling)
