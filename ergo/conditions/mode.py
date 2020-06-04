import jax.numpy as np

from ergo.scale import Scale

from . import condition


class ModeCondition(condition.Condition):
    """
    The specified outcome should be as close to being the most likely as possible.
    """

    outcome: float
    weight: float = 1.0

    def __init__(self, outcome, weight=1.0):
        self.outcome = outcome
        super().__init__(weight)

    def loss(self, dist) -> float:
        p_outcome = dist.pdf(self.outcome)
        p_highest = np.max(dist.ps)
        return self.weight * (p_highest - p_outcome) ** 2

    def _describe_fit(self, dist):
        description = super()._describe_fit(dist)
        description["p_outcome"] = dist.pdf(self.outcome)
        description["p_highest"] = np.max(dist.ps)
        return description

    def normalize(self, scale_min: float, scale_max: float):
        scale = Scale(scale_min, scale_max)
        normalized_outcome = scale.normalize_point(self.outcome)
        return self.__class__(normalized_outcome, self.weight)

    def denormalize(self, scale_min: float, scale_max: float):
        scale = Scale(scale_min, scale_max)
        denormalized_outcome = scale.denormalize_point(self.outcome)
        return self.__class__(denormalized_outcome, self.weight)

    def destructure(self):
        return (ModeCondition, (self.outcome, self.weight))

    def __str__(self):
        return f"The most likely outcome is {self.outcome}."
