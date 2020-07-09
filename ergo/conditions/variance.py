import jax.numpy as np

from ergo.scale import Scale

from . import condition


class VarianceCondition(condition.Condition):
    """
    The distribution should have as close to the specified variance as possible.
    """

    variance: float
    weight: float = 1.0

    def __init__(self, variance, weight=1.0):
        self.variance = variance
        super().__init__(weight)

    def loss(self, dist) -> float:
        return self.weight * (np.log(dist.variance()) - np.log(self.variance)) ** 2

    def _describe_fit(self, dist):
        description = super()._describe_fit(dist)
        description["variance"] = self.actual_variance(dist)
        return description

    def normalize(self, scale: Scale):
        normalized_variance = scale.normalize_variance(self.variance)
        return self.__class__(normalized_variance, self.weight)

    def denormalize(self, scale: Scale):
        denormalized_variance = scale.denormalize_point(self.variance)
        return self.__class__(denormalized_variance, self.weight)

    def destructure(self):
        return ((VarianceCondition,), (self.variance, self.weight))

    def __str__(self):
        return f"The variance is {self.variance}."

    def __repr__(self):
        return f"VarianceCondition(mean={self.variance}, weight={self.weight})"
