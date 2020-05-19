from dataclasses import dataclass

import scipy as oscipy

from .distribution import Distribution
from .types import Histogram


@dataclass
class HistogramDist(Distribution):
    """
    Create a continuous distribution from a density of its histogram
    """

    histogram: Histogram

    def __init__(self, histogram):
        self.histogram = histogram
        pairs = sorted([(v["x"], v["density"]) for v in self.histogram])
        xs = [x for (x, density) in pairs]
        bin_size = xs[1] - xs[0]
        bins = [x - bin_size / 2 for x in xs] + [xs[-1] + bin_size]
        densities = [density for (x, density) in pairs]
        self._rv = oscipy.stats.rv_histogram((densities, bins))

    def rv(self):
        return self._rv()

    def ppf(self, q):
        return self.rv().ppf(q)

    def cdf(self, x):
        return self.rv().cdf(x)

    def sample(self):
        return self.rv().sample()
