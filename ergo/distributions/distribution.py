"""
Base Distribution Class

Specifies interface for specific Distribution Classes
"""

from abc import ABC, abstractmethod

from .conditions import IntervalCondition


class Distribution(ABC):
    @abstractmethod
    def rv(self,):
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

    def percentiles(self, percentiles=None):
        if percentiles is None:
            percentiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        values = [self.ppf(q) for q in percentiles]
        return [
            IntervalCondition(percentile, max=value)
            for (percentile, value) in zip(percentiles, values)
        ]
