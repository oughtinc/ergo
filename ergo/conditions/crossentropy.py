import jax.numpy as np

from ergo.distributions import histogram

from . import condition


class CrossEntropyCondition(condition.Condition):
    p_dist: "histogram.HistogramDist"
    weight: float = 1.0

    def __init__(self, p_dist, weight=1.0):
        self.p_dist = p_dist
        super().__init__(weight)

    def loss(self, q_dist) -> float:
        return self.weight * self.p_dist.cross_entropy(q_dist)

    def destructure(self):
        return (CrossEntropyCondition, (np.array(self.p_dist.logps), self.weight))

    @classmethod
    def structure(cls, params):
        return cls(histogram.HistogramDist(params[0], traceable=True), params[1])

    def __str__(self):
        return "Minimize the cross-entropy of the two distributions"
