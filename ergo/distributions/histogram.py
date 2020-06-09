from dataclasses import dataclass

from jax import nn
import jax.numpy as np
import numpy as onp

from ergo import conditions, scale

from .distribution import Distribution
from .optimizable import Optimizable


@dataclass
class HistogramDist(Distribution, Optimizable):
    logps: np.DeviceArray

    def __init__(
        self, logps=None, scale_min=0, scale_max=1, traceable=False, direct_init=None
    ):
        # We assume that bin sizes are all equal
        if direct_init:
            self.logps = direct_init["logps"]
            self.ps = direct_init["ps"]
            self.cum_ps = direct_init["cum_ps"]
            self.bins = direct_init["bins"]
            self.size = direct_init["size"]
            self.scale_min = direct_init["scale_min"]
            self.scale_max = direct_init["scale_max"]
        else:
            init_numpy = np if traceable else onp
            self.logps = logps
            self.ps = np.exp(logps)
            self.cum_ps = np.array(init_numpy.cumsum(self.ps))
            self.bins = np.linspace(scale_min, scale_max, logps.size + 1)
            self.size = logps.size
            self.scale_min = scale_min
            self.scale_max = scale_max
        self.scale = scale.Scale(self.scale_min, self.scale_max)
        self.bin_size = (self.scale_max - self.scale_min) / self.logps.size

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, conditions.Condition):
            return self.__key() == other.__key()
        return NotImplemented

    def __key(self):
        return tuple(self.logps)

    def entropy(self):
        return -np.dot(self.ps, self.logps)

    def cross_entropy(self, q_dist):
        # Commented out to support Jax tracing:
        # assert self.scale_min == q_dist.scale_min, (self.scale_min, q_dist.scale_min)
        # assert self.scale_max == q_dist.scale_max
        # assert self.size == q_dist.size, (self.size, q_dist.size)
        return -np.dot(self.ps, q_dist.logps)

    def logpdf(self, x):
        return np.log(self.pdf(x))

    @classmethod
    def from_params(cls, fixed_params, opt_params, traceable=False):
        logps = nn.log_softmax(opt_params)
        return cls(logps, traceable=traceable)

    @staticmethod
    def initialize_optimizable_params(fixed_params):
        num_bins = fixed_params.get("num_bins", 100)
        return onp.full(num_bins, -num_bins)

    def normalize(self):
        return HistogramDist(self.logps, 0, 1)

    def denormalize(self, scale_min, scale_max):
        return HistogramDist(self.logps, scale_min, scale_max)

    def pdf(self, x):
        """
        If x is out of distribution range, returns 0. Otherwise returns the
        density at the lowest bin for which the upper bound of the bin
        is greater than or equal to x.

        :param x: The point in the distribution to get the density at
        """
        bin = np.maximum(np.argmax(self.bins >= x) - 1, 0)
        return np.where(
            (x < self.scale_min) | (x > self.scale_max),
            0,
            self.ps[bin] / self.bin_size,
        )

    def cdf(self, x):
        """
        If x is out of distribution range, returns 0/1. Otherwise returns the
        cumulative density at the lowest bin for which the upper bound of the bin
        is greater than or equal to x.

        :param x: The point in the distribution to get the cumulative density at
        """
        bin = np.maximum(np.argmax(self.bins >= x) - 1, 0)
        return np.where(
            x < self.scale_min, 0, np.where(x > self.scale_max, 1, self.cum_ps[bin],),
        )

    def ppf(self, q):
        return self.scale.denormalize_point(
            np.argmax(self.cum_ps >= np.minimum(q, self.cum_ps[-1])) / self.cum_ps.size
        )

    def sample(self):
        raise NotImplementedError

    def rv(self):
        raise NotImplementedError

    def destructure(self):
        return (
            HistogramDist,
            (
                self.logps,
                self.ps,
                self.cum_ps,
                self.bins,
                self.size,
                self.scale_min,
                self.scale_max,
            ),
        )

    @classmethod
    def structure(cls, params):
        return cls(
            direct_init={
                "logps": params[0],
                "ps": params[1],
                "cum_ps": params[2],
                "bins": params[3],
                "size": params[4],
                "scale_min": params[5],
                "scale_max": params[6],
            }
        )

    @classmethod
    def from_pairs(cls, pairs):
        sorted_pairs = sorted([(v["x"], v["density"]) for v in pairs])
        xs = [x for (x, density) in sorted_pairs]
        densities = [density for (x, density) in sorted_pairs]
        scale_min = xs[0]
        scale_max = xs[-1]
        logps = onp.log(onp.array(densities) / sum(densities))
        return cls(logps, scale_min=scale_min, scale_max=scale_max)

    def to_pairs(self):
        pairs = []
        bins = onp.array(self.bins)
        ps = onp.array(self.ps)
        for i, bin in enumerate(bins[:-1]):
            x = float((bin + bins[i + 1]) / 2.0)
            bin_size = float(bins[i + 1] - bin)
            density = float(ps[i]) / bin_size
            pairs.append({"x": x, "density": density})
        return pairs

    def to_lists(self):
        xs = []
        densities = []
        bins = onp.array(self.bins)
        ps = onp.array(self.ps)
        for i, bin in enumerate(bins[:-1]):
            x = float((bin + bins[i + 1]) / 2.0)
            bin_size = float(bins[i + 1] - bin)
            density = float(ps[i]) / bin_size
            xs.append(x)
            densities.append(density)
        return xs, densities

    def to_arrays(self):
        # TODO: vectorize
        xs, densities = self.to_lists()
        return np.array(xs), np.array(densities)
