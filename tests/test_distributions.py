import pytest

import ergo


def test_histogram_dist():
    histogram = [
        {"x": -0.00000001, "density": 0},
        {"x": 0, "density": 1},
        {"x": 0.2, "density": 1},
        {"x": 0.4, "density": 1},
        {"x": 0.6, "density": 1},
        {"x": 1, "density": 1},
        {"x": 1.00000001, "density": 0},
    ]
    dist = ergo.HistogramDist(histogram)
    for condition in dist.percentiles():
        assert condition.percentile == pytest.approx(condition.value, rel=0.01)
