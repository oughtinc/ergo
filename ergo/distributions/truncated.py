from dataclasses import dataclass
from .distribution import Distribution
import jax.numpy as np


def truncate(underlying_dist_class: Distribution, floor: float, ceiling: float):
    @dataclass
    class TruncatedDist(Distribution):
        underlying_dist: Distribution

        def ppf(self, q):
            raise NotImplementedError

        def pdf1(self, x):
            p_below = self.underlying_dist.cdf(floor)
            p_above = 1 - self.underlying_dist.cdf(ceiling)
            p_inside = 1 - (p_below + p_above)

            p_at_x = self.underlying_dist.pdf1(x) * (1 / p_inside)

            return np.where(x < floor, 0, np.where(x > ceiling, 0, p_at_x))

        @classmethod
        def from_params(cls, params):
            underlying_dist = underlying_dist_class.from_params(params)
            return cls(underlying_dist)

        @classmethod
        def from_conditions(cls, **kwargs):
            underlying_dist = underlying_dist_class.from_conditions(  # type: ignore
                containing_class=cls, scale_min=floor, scale_max=ceiling, **kwargs
            )
            return cls(underlying_dist)

    return TruncatedDist
