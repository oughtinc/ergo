"""
Base Distribution Class

Specifies interface for specific Distribution Classes
"""

from abc import ABC, abstractmethod

from ergo.scale import Scale


class Distribution(ABC):
    @abstractmethod
    def pdf(self, x):
        ...

    @abstractmethod
    def cdf(self, x):
        ...

    @abstractmethod
    def ppf(self, q):
        ...

    @abstractmethod
    def sample(self):
        ...

    @abstractmethod
    def normalize(self):
        ...

    @abstractmethod
    def denormalize(self, scale: Scale):
        ...

    def percentiles(self, percentiles=None):
        from ergo.conditions import IntervalCondition

        if percentiles is None:
            percentiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        values = [self.ppf(q) for q in percentiles]
        return [
            IntervalCondition(percentile, max=float(value))
            for (percentile, value) in zip(percentiles, values)
        ]
