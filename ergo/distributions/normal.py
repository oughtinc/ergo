"""
Normal distributions

Stub to be filled in
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from jax import jit, scipy
import jax.numpy as np
import numpy as onp
import scipy as oscipy

from .conditions import Condition
from .distribution import Distribution


@dataclass
class Normal(Distribution):
    loc: float
    scale: float
    metadata: Optional[Dict[str, Any]]

    def __init__(self, loc: float, scale: float, metadata=None):
        self.loc = loc
        self.scale = np.max([scale, 0.0000001])  # Do not allow values <= 0
        self.metadata = metadata

    def rv(self,):
        return oscipy.stats.norm(loc=self.loc, scale=self.scale)

    def sample(self):
        # FIXME (#296): This needs to be compatible with ergo sampling
        return onp.random.logistic(loc=self.loc, scale=self.scale)

    def cdf(self, x):
        y = (x - self.loc) / self.scale
        return scipy.stats.logistic.cdf(y)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf) at q.
        """
        return self.rv().ppf(q)

    @classmethod
    def from_samples_scipy(cls, samples) -> "Normal":
        with onp.errstate(all="raise"):  # type: ignore
            loc, scale = oscipy.stats.norm.fit(samples)
            return cls(loc, scale)

    @classmethod
    def from_conditions(
        self, conditions: List[Condition], initial_dist: Optional["Normal"] = None
    ):
        raise NotImplementedError

    @staticmethod
    def from_samples(samples) -> "Normal":
        raise NotImplementedError

    @staticmethod
    @jit
    def params_logpdf(x, loc, scale):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logistic.html
        y = (x - loc) / scale
        return scipy.stats.n.logpdf(y) - np.log(scale)
