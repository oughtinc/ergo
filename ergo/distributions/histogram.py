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
        self.scale = scale.Scale(scale_min, scale_max)

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

    def normalize(self, true_scale: scale.Scale = None):  # type: ignore
        """
        Normalize the histogram onto [0,1]
        Setting a true_scale allows you to express that the histogram to be normalized
        did not cover the entire scale of interest.
        E.g. -- imagine my histogram only has probability mass from 2 to 5,
        but I'm interested in p over [0,10].
        So I'll use Scale(0,10), and I'll get back a histogram with additional
        bins with 0 probability to cover the area from 0 to 2 and 5 to 10,
        where my histogram doesn't place any probability.
        :param true_scale: the full scale that I'm interested in probability over.
        """
        # if true_scale is not provided, assume that the histogram has
        # entries over the entire scale of interest
        if not true_scale:
            return HistogramDist(self.logps, 0, 1)

        if true_scale.scale_min is None or true_scale.scale_max is None:
            raise ValueError(
                "If you provide a true_scale, you must provide both a scale_min and a scale_max"
            )

        if (
            true_scale.scale_min > self.scale_min
            or true_scale.scale_max < self.scale_max
        ):
            raise ValueError(
                "Can only rescale hist to a scale that includes all of its current scale"
            )

        x_range_below = scale.Scale(true_scale.scale_min, self.scale_min)
        x_range_below_per_hist_range = x_range_below.range / self.scale.range
        num_x_bins_below = round(self.size * x_range_below_per_hist_range)

        x_range_above = scale.Scale(self.scale_max, true_scale.scale_max)
        x_range_above_per_hist_range = x_range_above.range / self.scale.range
        num_x_bins_above = round(self.size * x_range_above_per_hist_range)

        bins_below = onp.full(num_x_bins_below, float("-inf"))

        bins_above = onp.full(num_x_bins_above, float("-inf"))

        logps = onp.concatenate((bins_below, self.logps, bins_above))

        return HistogramDist(logps, 0, 1)

    def denormalize(self, scale_min, scale_max):
        return HistogramDist(self.logps, scale_min, scale_max)

    def pdf(self, x):
        """
        If x is out of distribution range, returns 0. Otherwise returns the
        density at the lowest bin for which the upper bound of the bin
        is greater than or equal to x.

        :param x: The point in the distribution to get the density at
        """
        return np.where(
            (x < self.scale_min) | (x > self.scale_max),
            0,
            self.ps[np.maximum(np.argmax(self.bins >= x) - 1, 0)],
        )

    def cdf(self, x):
        """
        If x is out of distribution range, returns 0/1. Otherwise returns the
        cumulative density at the lowest bin for which the upper bound of the bin
        is greater than or equal to x.

        :param x: The point in the distribution to get the cumulative density at
        """
        return np.where(
            x < self.scale_min,
            0,
            np.where(
                x > self.scale_max,
                1,
                self.cum_ps[np.maximum(np.argmax(self.bins >= x) - 1, 0)],
            ),
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

    def to_pairs(self, total_p: float = 1.0):
        """
        Represent the distribution as a list of pairs.
        Sometimes we use a histogram to represent
        just part of a probability distribution
        (if the rest will be represented by some other distribution)
        In this case, when converting to pairs,
        downscale the p in this distribution to some amount of total p.
        :total_p: the amount of p to keep in this distribution
        :return: a list of pairs representing the distribution
        """
        if total_p > 1:
            raise ValueError(
                "Can only scale down the distribution below total p of 1. Doesn't make sense for total p to be more than 1."
            )

        pairs = []
        bins = onp.array(self.bins)
        ps = onp.array(self.ps)
        for i, bin in enumerate(bins[:-1]):
            x = float((bin + bins[i + 1]) / 2.0)
            bin_size = float(bins[i + 1] - bin)
            density = (float(ps[i]) * total_p) / bin_size
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
