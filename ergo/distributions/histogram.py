from dataclasses import dataclass

from jax import nn
import jax.numpy as np
import numpy as onp

from ergo import conditions
from ergo.scale import Scale

from .distribution import Distribution
from .optimizable import Optimizable


@dataclass
class HistogramDist(Distribution, Optimizable):
    logps: np.DeviceArray

    def __init__(
        self,
        logps=None,
        scale=None,
        normed_bins=None,
        true_bins=None,
        traceable=False,
        direct_init=None,
    ):
        if direct_init:
            self.logps = direct_init["logps"]
            self.ps = direct_init["ps"]
            self.cum_ps = direct_init["cum_ps"]
            self.normed_bins = direct_init["normed_bins"]
            self.size = direct_init["size"]
            self.scale = direct_init["scale"]
            self.true_bins = None  # self.scale.denormalize_points(self.normed_bins)
        else:
            init_numpy = np if traceable else onp
            self.logps = logps
            self.ps = np.exp(logps)
            self.cum_ps = np.array(init_numpy.cumsum(self.ps))
            self.size = logps.size
            self.scale = scale if scale else Scale(0, 1)

            if true_bins:
                self.true_bins = true_bins
                self.normed_bins = self.scale.normalize_points(self.true_bins)
            elif normed_bins:
                self.normed_bins = normed_bins
                self.true_bins = self.scale.denormalize_points(self.normed_bins)
            else:
                print(
                    "No bin information provided, assuming probabilities correspond to a linear spacing in [0,1]"
                )
                self.normed_bins = np.linspace(0, 1, self.logps.size)
                self.true_bins = self.scale.denormalize_points(self.normed_bins)

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

    def normalize(self, scale: Scale = None):
        scale = scale if scale else Scale(0, 1)
        return HistogramDist(self.logps, scale=scale)

    def denormalize(self, scale: Scale):
        return HistogramDist(self.logps, scale)

    def pdf(self, true_x):
        """
        If x is out of distribution range, returns 0. Otherwise returns the
        density at the lowest bin for which the upper bound of the bin
        is greater than or equal to x.

        :param x: The point in the distribution to get the density at
        """
        x = self.scale.normalize_point(true_x)
        return np.where(
            (x < 0) | (x > 1),
            0,
            self.ps[np.maximum(np.argmax(self.normed_bins >= x) - 1, 0)],
        )

    def cdf(self, true_x):
        """
        If x is out of distribution range, returns 0/1. Otherwise returns the
        cumulative density at the lowest bin for which the upper bound of the bin
        is greater than or equal to x.

        :param x: The point in the distribution to get the cumulative density at
        """
        x = self.scale.normalize_point(true_x)
        return np.where(
            x < 0,
            0,
            np.where(
                x > 1,
                1,
                self.cum_ps[np.maximum(np.argmax(self.normed_bins >= x) - 1, 0)],
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
            (self.logps, self.ps, self.cum_ps, self.normed_bins, self.size,),
            *self.scale.destructure(),
        )

    @classmethod
    def structure(cls, *params):
        print(f"direct init bins are:\n {params[3]}")
        return cls(
            direct_init={
                "logps": params[0],
                "ps": params[1],
                "cum_ps": params[2],
                "normed_bins": params[3],
                "size": params[4],
                "scale": params[5](*params[6]),
            }
        )

    @classmethod
    def from_pairs(cls, pairs, scale: Scale, normalized=False):
        sorted_pairs = sorted([(v["x"], v["density"]) for v in pairs])
        xs = [x for (x, density) in sorted_pairs]
        densities = [density for (x, density) in sorted_pairs]
        logps = onp.log(
            onp.array(densities) / sum(densities)
        )  # TODO investigate why scale the densities with /sum(densities?)
        if normalized:
            return cls(logps, scale, normed_bins=xs)
        else:
            return cls(logps, scale, true_bins=xs)

    def to_pairs(self, normalized=False):
        pairs = []
        bins = self.normed_bins if normalized else self.true_bins
        ps = onp.array(self.ps)
        for i, bin in enumerate(bins[:-1]):
            x = float((bin + bins[i + 1]) / 2.0)
            density = float(ps[i])
            pairs.append({"x": x, "density": density})
        return pairs

    def to_pairs_normed(self, normalized=False):
        pairs = []
        bins = self.normed_bins if normalized else self.true_bins
        ps = onp.array(self.ps)
        auc = 0
        for i, bin in enumerate(bins[:-1]):
            x = float((bin + bins[i + 1]) / 2.0)
            bin_size = float(bins[i + 1] - bin)
            density = float(ps[i])
            auc += density * bin_size
            pairs.append({"x": x, "density": density})
        auc_norm = auc / len(self.ps)
        pairs = [{"x": x["x"], "density": x["density"] / auc_norm} for x in pairs]

    def to_pairs_points(self, normalized=False):
        pairs = []
        xs = self.normed_bins if normalized else self.true_bins
        ps = onp.array(self.ps)
        auc = 0
        from scipy.integrate import trapz  # type: ignore

        auc = trapz(ps, xs)
        auc_norm = auc / len(ps)

        for i, xs in enumerate(xs):
            pairs.append({"x": xs, "density": float(ps[i]) / auc_norm})

        import pandas as pd
        from scipy.integrate import trapz

        df = pd.DataFrame.from_records(pairs)
        print(df.iloc[:, 1].sum())
        inte = trapz(df.iloc[:, 1], df.iloc[:, 0])
        print(inte)

        return pairs

    def to_lists(self, normalized=False):
        xs = []
        densities = []
        bins = self.normed_bins if normalized else self.true_bins
        ps = onp.array(self.ps)
        for i, bin in enumerate(bins[:-1]):
            x = float((bin + bins[i + 1]) / 2.0)
            bin_size = float(bins[i + 1] - bin)
            density = float(ps[i]) / bin_size
            xs.append(x)
            densities.append(density)
        return xs, densities

    def to_arrays(self, normalized=False):
        # TODO: vectorize
        xs, densities = self.to_lists(normalized)
        return np.array(xs), np.array(densities)
