from jax import vmap
import jax.numpy as np

from ergo.distributions import histogram
from ergo.scale import Scale

from . import condition

# TODO: Implement normalize/denormalize for CrossEntropyCondition


class CrossEntropyCondition(condition.Condition):
    p_dist: "histogram.HistogramDist"
    weight: float = 1.0

    def __init__(self, p_dist, weight=1.0):
        self.p_dist = p_dist
        super().__init__(weight)

    def loss(self, q_dist) -> float:
        print(f'loss self.p_dist logps: {self.p_dist.logps}\n dist logps: {q_dist.logps})')
        return self.weight * self.p_dist.cross_entropy(q_dist)

    def destructure(self):
        return (CrossEntropyCondition, (np.array(self.p_dist.logps), self.weight))

    def _describe_fit(self, dist):
        description = super()._describe_fit(dist)
        print(f'self.p_dist logps: {self.p_dist.logps}\n dist logps: {dist.logps})')
        print(f'auc a: {np.sum(np.exp(self.p_dist.logps)) / self.p_dist.logps.size} auc b: {np.sum(np.exp(dist.logps)) / dist.logps.size}')
        print(description)
        return description

    @classmethod
    def structure(cls, params):
        return cls(histogram.HistogramDist(params[0], traceable=True), params[1])

    def __str__(self):
        return "Minimize the cross-entropy of the two distributions"


class PartialCrossEntropyCondition(condition.Condition):
    """
    Unlike CrossEntropyCondition, it's fine for (xs, ps) to
    only describe part of a distribution
    """

    xs: np.DeviceArray
    ps: np.DeviceArray
    weight: float = 1.0

    def __init__(self, xs, ps, weight):
        self.xs = xs
        self.ps = ps
        super().__init__(weight)

    def loss(self, q_dist) -> float:
        q_logps = vmap(q_dist.logpdf)(self.xs)
        cross_entropy = -np.dot(self.ps, q_logps)
        return self.weight * cross_entropy

    def destructure(self):
        return (PartialCrossEntropyCondition, (self.xs, self.ps, self.weight))

    @classmethod
    def structure(cls, params):
        return cls(params[0], params[1], params[2])

    def __str__(self):
        return "Minimize the cross-entropy of the two distributions (p may be partial)"

    def normalize(self, scale: Scale):
        # TODO: Vectorization should be part of what Scale does
        # scale.normalize_point / scale.denormalize_point is pretty much
        # vectorized anyway
        normed_xs = vmap(scale.normalize_point)(self.xs)
        return PartialCrossEntropyCondition(normed_xs, self.ps, self.weight)

    def denormalize(self, scale: Scale):
        # TODO: Vectorization should be part of what Scale does
        denormed_xs = vmap(scale.denormalize_point)(self.xs)
        return PartialCrossEntropyCondition(denormed_xs, self.ps, self.weight)
