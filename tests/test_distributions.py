import pytest

import ergo
from ergo.conditions import IntervalCondition, MaxEntropyCondition
from ergo.distributions.histogram import HistogramDist
from ergo.scale import Scale


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


def test_hist_pdf():
    uniform_dist = HistogramDist.from_conditions([MaxEntropyCondition()])

    # Off of scale
    assert uniform_dist.pdf(-0.5) == 0
    assert uniform_dist.pdf(1.5) == 0

    # Denormalized
    denormalized_dist = uniform_dist.denormalize(Scale(0, 2))
    assert denormalized_dist.pdf(1.5) != 0
    assert denormalized_dist.pdf(2.5) == 0


def test_hist_cdf():
    uniform_dist = HistogramDist.from_conditions([MaxEntropyCondition()])

    # Off of scale
    assert uniform_dist.cdf(-0.5) == 0
    assert uniform_dist.cdf(1.5) == 1

    # Edges of scale
    assert uniform_dist.cdf(0.005) != uniform_dist.cdf(0.015)
    assert uniform_dist.cdf(0.985) != uniform_dist.cdf(0.995)

    # Denormalized
    denormalized_dist = uniform_dist.denormalize(Scale(0, 2))
    assert denormalized_dist.cdf(1) == pytest.approx(0.5, abs=0.01)
    assert denormalized_dist.cdf(1.5) != 0
    assert denormalized_dist.cdf(2.5) == 1


def test_hist_ppf():
    uniform_dist = HistogramDist.from_conditions([])

    # Ends of scale; second is approx since implemented as start of last bin
    assert uniform_dist.ppf(0) == 0
    assert uniform_dist.ppf(1) == pytest.approx(1, abs=0.01)
