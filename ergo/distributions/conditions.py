from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from typing_extensions import TypedDict


class Condition(ABC):
    @abstractmethod
    def loss(self, dist):
        ...


@dataclass
class PercentileCondition(Condition):
    percentile: float
    value: float

    def __init__(self, percentile, value):
        if not 0 <= percentile <= 1:
            raise ValueError(f"Percentile should be in 0..1, got {percentile}")
        self.percentile = percentile
        self.value = value

    def loss(self, dist):
        target_percentile = self.percentile
        actual_percentile = dist.cdf(self.value)
        return (actual_percentile - target_percentile) ** 2

    def __str__(self):
        return f"There is a {round(self.percentile * 100)}% chance that the value is <{self.value}"


HistogramEntry = TypedDict("HistogramEntry", {"x": float, "density": float})
Histogram = List[HistogramEntry]


@dataclass
class HistogramCondition(Condition):
    histogram: Histogram

    def loss(self, dist):
        total_loss = 0.0
        for entry in self.histogram:
            target_density = entry["density"]
            actual_density = dist.pdf1(entry["x"])
            total_loss += (actual_density - target_density) ** 2
        return total_loss / len(self.histogram)

    def __str__(self):
        return f"The probability density function looks similar to the provided density function."
