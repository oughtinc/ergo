from dataclasses import dataclass
from .distribution import Distribution
from .scale import Scale


@dataclass
class TruncatedDist(Distribution):
    underlying_class: Distribution

    def rv(self,):
        ...

    def cdf(self, x):
        ...

    def ppf(self, q):
        ...

    def sample(self):
        ...

    def from_conditions(conditions, scale: Scale, *args):
        ...
