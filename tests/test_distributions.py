import pytest

import ergo
from ergo.distributions.conditions import IntervalCondition
from ergo.distributions.histogram import HistogramDist


@pytest.mark.xfail(reason="New histogram dist doesn't interpolate")
def test_histogram_dist():
    histogram = [
        {"x": 0, "density": 1},
        {"x": 0.2, "density": 1},
        {"x": 0.4, "density": 1},
        {"x": 0.6, "density": 1},
        {"x": 1, "density": 1},
    ]
    dist = ergo.HistogramDist.from_pairs(histogram)
    for condition in dist.percentiles():
        assert condition.max == pytest.approx(condition.p, abs=0.01)


def test_hist_from_percentile():
    for value in [0.01, 0.1, 0.5, 0.9]:
        conditions = [IntervalCondition(p=0.5, max=value)]
        dist = HistogramDist.from_conditions(conditions)
        assert dist.ppf(0.5) == pytest.approx(value, abs=0.1)
