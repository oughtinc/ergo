"""
Logistic distribution
"""

from dataclasses import dataclass

from jax import scipy
import jax.numpy as np
import scipy as oscipy

from ergo.scale import Scale
import ergo.static as static

from .distribution import Distribution


@dataclass
class Logistic(Distribution):
    loc: float  # normalized loc
    s: float  # normalized scale
    scale: Scale
    dist = scipy.stats.logistic
    odist = oscipy.stats.logistic

    def __init__(self, loc: float, s: float, scale: Scale, metadata=None):
        # TODO (#303): Raise ValueError on scale < 0
        self.loc = scale.normalize_point(loc)
        self.s = np.max([s, 0.0000001]) / scale.scale_range
        self.scale = scale
        self.metadata = metadata
        self.true_s = s  # convenience field not used ergo internal
        self.true_loc = loc  # convenience field not used ergo internal

    def __repr__(self):
        return f"Logistic(scale={self.scale}, true_loc={self.true_loc}, true_s={self.true_s}, normed_loc={self.loc}, normed_s={self.s}, metadata={self.metadata})"

    def rv(self):
        # returns normed rv object
        return self.odist(loc=self.loc, scale=self.s)

    def pdf(self, x):
        return (
            np.exp(self.logpdf(self.scale.normalize_point(x))) / self.scale.scale_range
        )

    def logpdf(self, x):
        # assumes x is normalized
        return static.logistic_logpdf(x, self.loc, self.s)

    def cdf(self, x):
        y = (self.scale.normalize_point(x) - self.loc) / self.s
        return self.dist.cdf(y)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf) at q.
        """
        return self.scale.denormalize_point(self.rv().ppf(q))

    def sample(self):
        # FIXME (#296): This needs to be compatible with ergo sampling
        return self.scale.denormalize_point(self.odist.rvs(loc=self.loc, scale=self.s))

    # Note: this is not strictly necessary as the distribution params are
    # always stored in normalized form
    def normalize(self):
        """
        Return the normalized condition.

        :param scale: the true scale
        :return: the condition normalized to [0,1]
        """
        return self.__class__(self.loc, self.s, Scale(0, 1), self.metadata)

    # Note: only the scale is necessary to change as the distribution params are
    # always stored in normalized form.
    def denormalize(self, scale: Scale):
        """
        Assume that the distribution has been normalized to be over [0,1].
        Return the distribution on the true scale
        :param scale: the true-scale
        """
        denormalized_loc = scale.denormalize_point(self.loc)
        denormalized_s = self.s * scale.scale_range
        return self.__class__(denormalized_loc, denormalized_s, scale, self.metadata)
