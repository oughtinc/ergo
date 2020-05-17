from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax.numpy as np
import scipy as oscipy

from .distribution import Distribution


@dataclass
class DistributionLS(Distribution):
    loc: float
    scale: float
    metadata: Optional[Dict[str, Any]]
    dist: Any

    def __init__(self, loc: float, scale: float, dist, metadata=None):
        self.loc = loc
        self.scale = np.max([scale, 0.0000001])  # Do not allow values <= 0
        self.dist = dist
        self.metadata = metadata

    def __mul__(self, x):
        return self.__class__(self.loc * x, self.scale * x)

    def rv(self,):
        return self.dist(loc=self.loc, scale=self.scale)

    def sample(self):
        # FIXME (#296): This needs to be compatible with ergo sampling
        return self.dist(loc=self.loc, scale=self.scale)

    def cdf(self, x):
        y = (x - self.loc) / self.scale
        return self.cdf(y)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf) at q.
        """
        return self.rv().ppf(q)


@dataclass
class Logistic(DistributionLS):
    def __init__(self, loc: float, scale: float, metadata=None):
        super().__init__(loc, scale, oscipy.stats.logistic, metadata)


@dataclass
class Normal(DistributionLS):
    def __init__(self, loc: float, scale: float, metadata=None):
        super().__init__(loc, scale, oscipy.stats.norm, metadata)
