import jax.numpy as np

from ergo.distributions import histogram
import ergo.static as static

from . import condition


class WassersteinCondition(condition.Condition):
    p_dist: "histogram.HistogramDist"
    weight: float = 1.0

    def __init__(self, p_dist, weight=1.0):
        self.p_dist = p_dist
        super().__init__(weight)

    def loss(self, q_dist) -> float:
        return self.weight * static.wasserstein_distance(self.p_dist.ps, q_dist.ps)

    def destructure(self):
        return (WassersteinCondition, (np.array(self.p_dist.logprobs), self.weight))

    @classmethod
    def structure(cls, params):
        return cls(histogram.HistogramDist(params[0]), params[1], traceable=True)

    def __str__(self):
        return "Minimize the Wasserstein distance between the two distributions"
