from dataclasses import dataclass
from .distribution import Distribution
import jax.numpy as np


def truncate(underlying_dist_class: Distribution, floor: float, ceiling: float):
    """
    Get a version of the distribution passed in
    that's truncated to the given floor and ceiling

    :param underlying_dist_class: the distribution class to truncate
    :param floor: throw out probability mass below this value
    :param ceiling: throw out probability mass above this value
    :return: a truncated version of the distribution class
    that was passed in
    """

    @dataclass
    class TruncatedDist(Distribution):
        """
        A wrapper around the underlying distribution that throws out
        probability mass outside the range defined by the floor/ceiling
        and then renormalizes

        :param Distribution: the underlying distribution to truncate
        """

        underlying_dist: Distribution

        def __init__(self, underlying_dist, *args, **kwargs):
            self.underlying_dist = underlying_dist
            p_below = underlying_dist.cdf(floor)
            p_above = 1 - underlying_dist.cdf(ceiling)
            self.p_inside = 1 - (p_below + p_above)
            return super().__init__(*args, **kwargs)

        def ppf(self, q):
            raise NotImplementedError

        def pdf1(self, x):
            p_at_x = self.underlying_dist.pdf1(x) * (1 / self.p_inside)
            # this line shouldn't be necessary: in practice, we don't
            # expect to encounter xs outside the range.
            return np.where(x < floor, 0, np.where(x > ceiling, 0, p_at_x))

        def normalize(self, scale_min: float, scale_max: float):
            return self.underlying_dist.normalize(scale_min, scale_max)  # type: ignore

        @classmethod
        def from_params(cls, params):
            underlying_dist = underlying_dist_class.from_params(params)
            return cls(underlying_dist)

        @classmethod
        def from_conditions(cls, conditions, **kwargs):
            underlying_dist = underlying_dist_class.from_conditions(  # type: ignore
                containing_class=cls,
                scale_min=floor,
                scale_max=ceiling,
                conditions=conditions,
                **kwargs
            )
            return cls(underlying_dist)

    return TruncatedDist
