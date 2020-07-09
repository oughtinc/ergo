from typing import Optional

from ergo.scale import Scale

from . import condition


class IntervalCondition(condition.Condition):
    """
    The specified interval should include as close to the specified
    probability mass as possible

    :raises ValueError: max must be strictly greater than min
    """

    p: float
    min: Optional[float]
    max: Optional[float]
    weight: float

    def __init__(self, p, min=None, max=None, weight=1.0):
        self.p = p
        self.min = min
        self.max = max
        super().__init__(weight)

    def actual_p(self, dist) -> float:
        cdf_at_min = dist.cdf(self.min) if self.min is not None else 0
        cdf_at_max = dist.cdf(self.max) if self.max is not None else 1
        return cdf_at_max - cdf_at_min

    def loss(self, dist):
        actual_p = self.actual_p(dist)
        return self.weight * (actual_p - self.p) ** 2

    def _describe_fit(self, dist):
        description = super()._describe_fit(dist)
        description["p_in_interval"] = self.actual_p(dist)
        return description

    def normalize(self, scale: Scale):
        normalized_min = (
            scale.normalize_point(self.min) if self.min is not None else None
        )
        normalized_max = (
            scale.normalize_point(self.max) if self.max is not None else None
        )
        return self.__class__(self.p, normalized_min, normalized_max, self.weight)

    def denormalize(self, scale: Scale):
        denormalized_min = (
            scale.denormalize_point(self.min) if self.min is not None else None
        )
        denormalized_max = (
            scale.denormalize_point(self.max) if self.max is not None else None
        )
        return self.__class__(self.p, denormalized_min, denormalized_max, self.weight)

    def destructure(self):
        return ((IntervalCondition,), (self.p, self.min, self.max, self.weight))

    def shape_key(self):
        return (self.__class__.__name__, self.min is None, self.max is None)

    def __str__(self):
        return f"There is a {self.p:.0%} chance that the value is in [{self.min}, {self.max}]"

    def __repr__(self):
        return f"IntervalCondition(p={self.p}, min={self.min}, max={self.max}, weight={self.weight})"
