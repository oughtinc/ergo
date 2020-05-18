from abc import ABC, abstractmethod


"""
Base Distribution Class

Specifies interface for specific Distribution Classes
"""


class Distribution(ABC):
    @abstractmethod
    def __mul__(self, x):
        ...

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
