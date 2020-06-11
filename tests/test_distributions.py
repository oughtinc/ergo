import pytest

import ergo
from ergo.conditions import IntervalCondition, MaxEntropyCondition
from ergo.distributions.histogram import HistogramDist
from ergo.scale import Scale, LogScale

scales_to_test = [
    Scale(0, 1),
    Scale(0, 10000),
    Scale(-1, 1),
    LogScale(0, 1, 10),
    LogScale(-1, 1, 10),
    LogScale(0, 1028, 2),
]


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


@pytest.mark.parametrize("scale", scales_to_test)
def test_hist_pdf(scale: Scale):
    uniform_dist = HistogramDist.from_conditions([MaxEntropyCondition()], scale=scale)

    assert uniform_dist.pdf(scale.denormalize_point(0.5)) != 0
    assert uniform_dist.pdf(scale.denormalize_point(1.5)) == 0


@pytest.mark.parametrize("scale", scales_to_test)
def test_hist_cdf(scale: Scale):
    uniform_dist = HistogramDist.from_conditions([MaxEntropyCondition()], scale=scale)

    # Off of scale
    assert uniform_dist.cdf(scale.denormalize_point(-0.5)) == 0
    assert uniform_dist.cdf(scale.denormalize_point(1.5)) == 1

    # Edges of scale
    assert uniform_dist.cdf(scale.denormalize_point(0.005)) != uniform_dist.cdf(
        scale.denormalize_point(0.015)
    )
    assert uniform_dist.cdf(scale.denormalize_point(0.985)) != uniform_dist.cdf(
        scale.denormalize_point(0.995)
    )

    # Denormalized onto a different scale
    denormalized_dist = uniform_dist.denormalize(Scale(0, 2))
    assert denormalized_dist.cdf(1) == pytest.approx(0.5, abs=0.01)
    assert denormalized_dist.cdf(1.5) != 0
    assert denormalized_dist.cdf(2.5) == 1


@pytest.mark.parametrize("scale", scales_to_test)
def test_hist_ppf(scale: Scale):
    uniform_dist = HistogramDist.from_conditions([], scale=scale)

    # Ends of scale; second is approx since implemented as start of last bin
    assert uniform_dist.ppf(scale.denormalize_point(0)) == scale.scale_min
    assert uniform_dist.ppf(scale.denormalize_point(1)) == pytest.approx(
        scale.scale_max, rel=0.1
    )
