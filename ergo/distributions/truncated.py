from dataclasses import dataclass

import jax.numpy as np

from .distribution import Distribution


def truncate(BaseDistClass):
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
    class GenericTruncatedDist:
        """
        A wrapper around the underlying distribution that throws out
        probability mass outside the range defined by the floor/ceiling
        and then renormalizes

        :param Distribution: the underlying distribution to truncate
        """

        base_dist: Distribution
        floor: float
        ceiling: float

        def __init__(self, base_dist, floor: float, ceiling: float, *args, **kwargs):
            self.base_dist = base_dist
            self.floor = floor
            self.ceiling = ceiling
            p_below = base_dist.cdf(floor)
            p_above = 1 - base_dist.cdf(ceiling)
            self.p_inside = 1 - (p_below + p_above)
            return super().__init__(*args, **kwargs)

        def pdf1(self, x):
            p_at_x = self.base_dist.pdf1(x) * (1 / self.p_inside)
            # this line shouldn't be necessary: in practice, we don't
            # expect to encounter xs outside the range.
            return np.where(x < self.floor, 0, np.where(x > self.ceiling, 0, p_at_x))

    return type("TruncatedDist", (BaseDistClass, GenericTruncatedDist), {})
