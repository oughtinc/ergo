from dataclasses import dataclass
from typing import List

from jax import grad, jit, nn
import jax.numpy as np
import numpy as onp
import scipy as oscipy

from . import conditions, distribution


@dataclass
class HistogramDist(distribution.Distribution):
    logps: np.DeviceArray

    def __init__(self, logps, scale_min=0, scale_max=1, traceable=False):
        init_numpy = np if traceable else onp
        self.logps = logps
        self.ps = np.exp(logps)
        self.cum_ps = np.array(init_numpy.cumsum(self.ps))
        self.bins = np.linspace(scale_min, scale_max, logps.size + 1)
        self.size = logps.size
        self.scale_min = scale_min
        self.scale_max = scale_max

    def entropy(self):
        return -np.dot(self.ps, self.logps)

    def cross_entropy(self, q_dist):
        assert self.scale_min == q_dist.scale_min, (self.scale_min, q_dist.scale_min)
        assert self.scale_max == q_dist.scale_max
        assert self.size == q_dist.size, (self.size, q_dist.size)
        return -np.dot(self.ps, q_dist.logps)

    def pdf(self, x):
        return self.ps[self.bins > x][0]

    def cdf(self, x):
        return self.cum_ps[self.bins > x][0]

    def ppf(self, q):
        return np.where(self.cum_ps > q)[0][0] / self.cum_ps.size

    def sample(self):
        raise NotImplementedError

    def rv(self):
        raise NotImplementedError

    def normalize(self):
        return HistogramDist(self.logps, 0, 1)

    def denormalize(self, scale_min, scale_max):
        return HistogramDist(self.logps, scale_min, scale_max)

    @classmethod
    def from_conditions(
        cls,
        conditions: List["conditions.Condition"],
        scale_min=0,
        scale_max=1,
        num_bins=100,
        verbose=False,
    ):
        normalized_conditions = [
            condition.normalize(scale_min, scale_max) for condition in conditions
        ]

        def loss(params):
            dist = cls.from_params(params, traceable=True)
            total_loss = sum(
                condition.loss(dist) for condition in normalized_conditions
            )
            return total_loss * 100

        loss = jit(loss)
        jac = jit(grad(loss))
        normalized_dist = cls.from_loss(loss=loss, jac=jac, num_bins=num_bins)

        if verbose:
            for condition in normalized_conditions:
                print(condition)
                print(condition.describe_fit(normalized_dist))

        return normalized_dist.denormalize(scale_min, scale_max)

    @classmethod
    def from_loss(cls, loss, jac, num_bins=100):
        x0 = cls.initialize_params(num_bins)
        results = oscipy.optimize.minimize(loss, jac=jac, x0=x0)
        return cls.from_params(results.x)

    @classmethod
    def from_params(cls, params, traceable=False):
        logps = nn.log_softmax(params)
        return cls(logps, traceable=traceable)

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

    @staticmethod
    def initialize_params(num_bins):
        return onp.full(num_bins, -num_bins)
