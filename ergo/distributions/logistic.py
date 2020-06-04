"""
Logistic distribution
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from jax import scipy
import jax.numpy as np
import scipy as oscipy

from ergo.scale import Scale
import ergo.static as static

from .distribution import Distribution


@dataclass
class Logistic(Distribution):
    loc: float
    scale: float
    metadata: Optional[Dict[str, Any]]

    dist = scipy.stats.logistic
    odist = oscipy.stats.logistic

    def __init__(self, loc: float, scale: float, metadata=None):
        # TODO (#303): Raise ValueError on scale < 0
        self.scale = np.max([scale, 0.0000001])
        self.loc = loc
        self.metadata = metadata

    def rv(self):
        return self.odist(loc=self.loc, scale=self.scale)

    def logpdf(self, x):
        return static.logistic_logpdf(x, self.loc, self.scale)

    def cdf(self, x):
        y = (x - self.loc) / self.scale
        return self.dist.cdf(y)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf) at q.
        """
        return self.rv().ppf(q)

    def sample(self):
        # FIXME (#296): This needs to be compatible with ergo sampling
        return self.odist.rvs(loc=self.loc, scale=self.scale)

    def normalize(self, scale_min: float, scale_max: float):
        """
        Assume that the condition's true range is [scale_min, scale_max].
        Return the normalized condition.

        :param scale_min: the true-scale minimum of the range
        :param scale_max: the true-scale maximum of the range
        :return: the condition normalized to [0,1]
        """
        scale = Scale(scale_min, scale_max)
        normalized_loc = scale.normalize_point(self.loc)
        normalized_scale = self.scale / scale.range
        return self.__class__(normalized_loc, normalized_scale, self.metadata)

    def denormalize(self, scale_min: float, scale_max: float):
        """
        Assume that the distribution has been normalized to be over [0,1].
        Return the distribution on the true scale of [scale_min, scale_max]

        :param scale_min: the true-scale minimum of the range
        :param scale_max: the true-scale maximum of the range
        """
        scale = Scale(scale_min, scale_max)
        denormalized_loc = scale.denormalize_point(self.loc)
        denormalized_scale = self.scale * scale.range
        return self.__class__(denormalized_loc, denormalized_scale, self.metadata)
