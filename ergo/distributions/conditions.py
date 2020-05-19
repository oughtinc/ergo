from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from typing_extensions import TypedDict


class Condition(ABC):
    @abstractmethod
    def loss(self, dist):
        ...


@dataclass
class PercentileCondition(Condition):
    percentile: float
    value: float
    weight: float

    def __init__(self, percentile, value, weight=1.0):
        if not 0 <= percentile <= 1:
            raise ValueError(f"Percentile should be in 0..1, got {percentile}")
        self.percentile = percentile
        self.value = value
        self.weight = weight

    def loss(self, dist):
        target_percentile = self.percentile
        actual_percentile = dist.cdf(self.value)
        return self.weight * (actual_percentile - target_percentile) ** 2

    def __str__(self):
        return (
            f"There is a {self.percentile:.0%} chance that the value is <{self.value}"
        )


HistogramEntry = TypedDict("HistogramEntry", {"x": float, "density": float})
Histogram = List[HistogramEntry]


@dataclass
class HistogramCondition(Condition):
    histogram: Histogram
    weight: float = 1.0

    def loss(self, dist):
        total_loss = 0.0
        for entry in self.histogram:
            target_density = entry["density"]
            actual_density = dist.pdf1(entry["x"])
            total_loss += (actual_density - target_density) ** 2
        return self.weight * total_loss / len(self.histogram)

    def __str__(self):
        return f"The probability density function looks similar to the provided density function."


@dataclass
class IntervalCondition(Condition):
    p: float
    low: Optional[float]
    high: Optional[float]
    weight: float

    def __init__(self, p, low=None, high=None, weight=1.0):
        if low is not None and high is not None and high <= low:
            raise ValueError(
                f"High must be greater than low, got high: {high}, low: {low}"
            )

        self.p = p
        self.low = low
        self.high = high
        self.weight = weight

    def loss(self, dist):
        cdf_at_low = dist.cdf(self.low) if self.low is not None else 0
        cdf_at_high = dist.cdf(self.high) if self.high is not None else 1
        actual_p = cdf_at_high - cdf_at_low
        return self.weight * (actual_p - self.p) ** 2

    def __str__(self):
        return f"There is a {self.p:.0%} chance that the value is in [{self.low}, {self.high}]"
