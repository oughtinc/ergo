from dataclasses import dataclass

from jax import nn
import jax.numpy as np
import numpy as onp
from scipy.integrate import trapz
from scipy.interpolate import interp1d

from ergo import conditions
from ergo.scale import LogScale, Scale

from .distribution import Distribution
from .optimizable import Optimizable


@dataclass
class HistogramDist(Distribution, Optimizable):
    logps: np.DeviceArray

    def __init__(
        self, logps=None, scale=None, traceable=False, direct_init=None,
    ):
        # We assume that bin sizes are all equal
        if direct_init:
            self.logps = direct_init["logps"]
            self.ps = direct_init["ps"]
            self.cum_ps = direct_init["cum_ps"]
            self.bins = direct_init["bins"]
            self.size = direct_init["size"]
            self.scale = direct_init["scale"]
        else:
            init_numpy = np if traceable else onp
            self.logps = logps
            self.ps = np.exp(logps)
            self.cum_ps = np.array(init_numpy.cumsum(self.ps))
            self.size = logps.size
            self.scale = scale if scale else Scale(0, 1)
            self.bins = np.linspace(0, 1, self.logps.size + 1)
        self.truebin_size = (self.scale.high - self.scale.low) / self.logps.size

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, conditions.Condition):
            return self.__key() == other.__key()
        return NotImplemented

    def __key(self):
        return tuple(self.logps)

    def __repr__(self):
        return f"{self.__class__}(size={self.size}, scale={self.scale}, bins={self.bins}, ps: {self.ps})"

    def entropy(self):
        return -np.dot(self.ps, self.logps)

    def cross_entropy(self, q_dist):
        # Commented out to support Jax tracing:
        # assert self.low == q_dist.low, (self.low, q_dist.low)
        # assert self.high == q_dist.high
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
        return onp.full(num_bins, -float(num_bins))

    def normalize(self):
        return HistogramDist(self.logps, scale=Scale(0, 1))

    def denormalize(self, scale: Scale):
        return HistogramDist(self.logps, scale)

    def pdf(self, x):
        """
        If x is out of distribution range, returns 0. Otherwise returns the
        density at the lowest bin for which the upper bound of the bin
        is greater than or equal to x.

        :param x: The point in the distribution to get the density at
        """
        x = self.scale.normalize_point(x)
        bin = np.maximum(np.argmax(self.bins >= x) - 1, 0)
        return np.where((x < 0) | (x > 1), 0, self.ps[bin] / self.truebin_size)

    def cdf(self, x):
        """
        If x is out of distribution range, returns 0/1. Otherwise returns the
        cumulative density at the lowest bin for which the upper bound of the bin
        is greater than or equal to x.

        :param x: The point in the distribution to get the cumulative density at
        """
        x = self.scale.normalize_point(x)
        bin = np.maximum(np.argmax(self.bins >= x) - 1, 0)
        return np.where(x < 0, 0, np.where(x > 1, 1, self.cum_ps[bin]))

    def ppf(self, q):
        return self.scale.denormalize_point(
            np.argmax(self.cum_ps >= np.minimum(q, self.cum_ps[-1])) / self.cum_ps.size
        )

    def sample(self):
        raise NotImplementedError

    def rv(self):
        raise NotImplementedError

    def destructure(self):
        scale_cls, scale_params = self.scale.destructure()
        return (
            (HistogramDist, scale_cls),
            (self.logps, self.ps, self.cum_ps, self.bins, self.size, scale_params),
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
                "scale": params[6].structure(params[5]),
            }
        )

    @classmethod
    def from_pairs(cls, pairs, scale: Scale, normalized=False, bins=201):
        sorted_pairs = sorted([(v["x"], v["density"]) for v in pairs])
        xs = [x for (x, density) in sorted_pairs]
        if not normalized:
            xs = scale.normalize_points(xs)
        densities = [density for (x, density) in sorted_pairs]

        bins = onp.linspace(0, 1, bins)
        target_xs = (bins[:-1] + bins[1:]) / 2  # get midpoint of each bin for x coord
        # interpolate ps at normalized x bins
        if not (
            len(xs) == len(target_xs) and np.isclose(xs, target_xs, rtol=1e-04).all()
        ):
            f = interp1d(xs, densities)
            densities = f(target_xs)
        logps = onp.log(onp.array(densities) / sum(densities))
        return cls(logps, scale)

    def to_lists(self, true_scale=True, verbose=False):
        bins = self.bins
        xs = (bins[:-1] + bins[1:]) / 2

        if true_scale:
            xs = np.array(self.scale.denormalize_points(xs))
            bins = np.array(self.scale.denormalize_points(self.bins))

        if type(self.scale) != LogScale:
            ps = np.divide(self.ps, bins[1:] - bins[:-1])

        else:
            auc = trapz(self.ps, xs)
            ps = self.ps / auc

        if verbose:
            import pandas as pd

            df = pd.DataFrame(data={"x": xs, "density": ps})
            auc = trapz(df["density"], df["x"])
            print(f"AUC is {auc}")
            print(f"scale is {self.scale}")

        return xs, ps

    def to_pairs(
        self, true_scale=True, verbose=False,
    ):

        xs, ps = self.to_lists(true_scale=True, verbose=False)

        return [
            {"x": float(x), "density": float(density)} for x, density in zip(xs, ps)
        ]

    def to_arrays(self, normalized=False):
        # TODO: vectorize
        xs, densities = self.to_lists(normalized)
        return np.array(xs), np.array(densities)
