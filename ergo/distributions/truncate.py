from dataclasses import dataclass
from typing import Type, TypeVar

import jax.numpy as np

from .distribution import Distribution

D = TypeVar("D", bound=Distribution)


def truncate(DistClass: Type[D], floor: float = 0.0, ceiling: float = 1.0):
    @dataclass
    class TruncatedDist(DistClass):  # type: ignore
        def __post_init__(self):
            p_below = super().cdf(floor)
            p_above = 1 - super().cdf(ceiling)
            self.p_inside = 1 - (p_below + p_above)
            self.logp_inside = np.log(self.p_inside)
            print(p_below, self.p_inside, p_above)

        def logpdf(self, x):
            logp_x = super().logpdf(x) - self.logp_inside
            res = np.where(x < floor, -np.inf, np.where(x > ceiling, -np.inf, logp_x))
            return res

        def cdf(self, x):
            cdf_x = super().cdf(x) / self.p_inside
            return np.where(x < floor, 0, np.where(x > ceiling, 1, cdf_x))

        def ppf(self, q):
            raise NotImplementedError

        def sample(self):
            raise NotImplementedError

    return TruncatedDist
