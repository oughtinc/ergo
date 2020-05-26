from dataclasses import dataclass
from typing import List

from jax import grad, jit, nn
import jax.numpy as np
import scipy as oscipy

from .conditions import Condition
from .distribution import Distribution


@dataclass
class HistogramDist(Distribution):
    logps: np.DeviceArray

    def __init__(self, logps, scale_min=0, scale_max=1):
        self.logps = logps  # probability of each bin..
        self.ps = np.exp(logps)
        self.cum_ps = np.cumsum(self.ps)
        self.bins = np.linspace(scale_min, scale_max, logps.size + 1)
        self.size = logps.size
        self.scale_min = scale_min
        self.scale_max = scale_max

    def entropy(self):
        return -np.dot(self.ps, self.logps)

    def cross_entropy(self, q_dist):
        assert self.scale_min == q_dist.scale_min
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

    def normalize(self, scale_min, scale_max):
        return HistogramDist(self.logps, 0, 1)

    def denormalize(self, scale_min, scale_max):
        return HistogramDist(self.logps, scale_min, scale_max)

    @classmethod
    def from_conditions(
        cls,
        conditions: List[Condition],
        scale_min=0,
        scale_max=1,
        num_bins=100,
        verbose=False,
    ):
        normalized_conditions = [
            condition.normalize(scale_min, scale_max) for condition in conditions
        ]

        def loss(params):
            dist = cls.from_params(params)
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
    def from_params(cls, params):
        logps = nn.log_softmax(params)
        return cls(logps)

    @classmethod
    def from_pairs(cls, pairs):
        sorted_pairs = sorted([(v["x"], v["density"]) for v in pairs])
        xs = [x for (x, density) in sorted_pairs]
        densities = [density for (x, density) in sorted_pairs]
        scale_min = xs[0]
        scale_max = xs[-1]
        logps = np.log(np.array(densities) / sum(densities))
        return cls(logps, scale_min=scale_min, scale_max=scale_max)

    def to_pairs(self):
        pairs = []
        for i, bin in enumerate(self.bins[:-1]):
            x = float((bin + self.bins[i + 1]) / 2.0)
            density = float(self.ps[i])
            pairs.append({"x": x, "density": density})
        return pairs

    @staticmethod
    def initialize_params(num_bins):
        return np.full(num_bins, -num_bins)
