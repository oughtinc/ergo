import pytest

from ergo import Logistic, LogisticMixture
from ergo.distributions.conditions import (
    HistogramCondition,
    PercentileCondition,
    IntervalCondition,
)
from dataclasses import dataclass


def test_mixture_from_percentile():
    for value in [0.01, 0.1, 1, 3]:
        conditions = [PercentileCondition(percentile=0.5, value=value)]
        dist = LogisticMixture.from_conditions(
            conditions, num_components=1, verbose=True
        )
        loc = dist.components[0].loc
        assert loc == pytest.approx(value, rel=0.1), loc


def test_mixture_from_percentiles():
    conditions = [
        PercentileCondition(percentile=0.1, value=1),
        PercentileCondition(percentile=0.5, value=2),
        PercentileCondition(percentile=0.6, value=3),
    ]
    dist = LogisticMixture.from_conditions(conditions, num_components=3, verbose=True)
    for condition in conditions:
        assert dist.cdf(condition.value) == pytest.approx(condition.percentile, rel=0.1)


def test_percentiles_from_mixture():
    mixture = LogisticMixture(
        components=[Logistic(loc=1, scale=0.1), Logistic(loc=2, scale=0.1)],
        probs=[0.5, 0.5],
    )
    conditions = mixture.percentiles(percentiles=[0.1, 0.5, 0.9])
    for condition in conditions:
        if condition.percentile == pytest.approx(0.5):
            assert condition.value == pytest.approx(1.5, rel=0.01)
    return conditions


def test_percentile_roundtrip():
    conditions = [
        PercentileCondition(0.01, 0.61081324517545),
        PercentileCondition(0.1, 0.8613634657212543),
        PercentileCondition(0.25, 1),
        PercentileCondition(0.5, 1.5),
        PercentileCondition(0.75, 2),
        PercentileCondition(0.9, 2.1386364698410034),
        PercentileCondition(0.99, 2.3891870975494385),
    ]
    mixture = LogisticMixture.from_conditions(
        conditions, num_components=3, verbose=True
    )
    recovered_conditions = mixture.percentiles(
        percentiles=[condition.percentile for condition in conditions]
    )
    for (condition, recovered_condition) in zip(conditions, recovered_conditions):
        assert recovered_condition.value == pytest.approx(condition.value, rel=0.1)


def test_mixture_from_histogram(histogram):
    conditions = [HistogramCondition(histogram)]
    mixture = LogisticMixture.from_conditions(
        conditions, num_components=3, verbose=True
    )
    for entry in histogram:
        assert mixture.pdf1(entry["x"]) == pytest.approx(entry["density"], abs=0.2)


def test_weights():
    conditions = [
        PercentileCondition(percentile=0.4, value=1, weight=0.01),
        PercentileCondition(percentile=0.5, value=2, weight=100),
        PercentileCondition(percentile=0.8, value=2.2, weight=0.01),
        PercentileCondition(percentile=0.9, value=2.3, weight=0.01),
    ]
    dist = LogisticMixture.from_conditions(conditions, num_components=1, verbose=True)
    assert dist.components[0].loc == pytest.approx(2, rel=0.1)


@dataclass
class Uniform:
    min: float = 0
    max: float = 1

    def cdf(self, value):
        if value < self.min:
            return 0

        if value > self.max:
            return 1

        return (value - self.min) / (self.max - self.min)


def test_interval_loss():
    dist = Uniform(min=-1, max=1)

    assert IntervalCondition(p=0.5, low=0, high=1).loss(dist) == 0
    assert IntervalCondition(p=0.25, low=0, high=1).loss(dist) == 0.25 ** 2
    assert IntervalCondition(p=1, high=0).loss(dist) == 0.5 ** 2
    assert IntervalCondition(p=1).loss(dist) == 0
    assert IntervalCondition(p=0, low=-1, high=1).loss(dist) == 1
    assert IntervalCondition(p=0, low=-1, high=1, weight=10).loss(dist) == 10


def test_histogram_condition(histogram, normalized_logistic_mixture):
    condition = HistogramCondition(histogram)
    vectorized_loss = float(condition.loss(normalized_logistic_mixture))
    loop_loss = float(condition._loss_loop(normalized_logistic_mixture))
    assert vectorized_loss == pytest.approx(loop_loss, rel=0.0001)


def test_mixed_1(histogram):
    conditions = [
        PercentileCondition(percentile=0.4, value=1),
        PercentileCondition(percentile=0.45, value=1.2),
        PercentileCondition(percentile=0.47, value=1.3),
        PercentileCondition(percentile=0.5, value=2),
        PercentileCondition(percentile=0.8, value=2.2),
        PercentileCondition(percentile=0.9, value=2.3),
        HistogramCondition(histogram),
    ]
    dist = LogisticMixture.from_conditions(conditions, num_components=3, verbose=True)
    assert dist.pdf1(-5) == pytest.approx(0, abs=0.1)
    assert dist.pdf1(6) == pytest.approx(0, abs=0.1)


def test_mixed_2(histogram):
    conditions = [
        HistogramCondition(histogram),
        PercentileCondition(percentile=0.4, value=1),
        PercentileCondition(percentile=0.45, value=1.2),
        PercentileCondition(percentile=0.48, value=1.3),
        PercentileCondition(percentile=0.5, value=2),
        PercentileCondition(percentile=0.7, value=2.2),
        PercentileCondition(percentile=0.9, value=2.3),
    ]
    dist = LogisticMixture.from_conditions(conditions, num_components=3, verbose=True)
    assert dist.pdf1(-5) == pytest.approx(0, abs=0.1)
    assert dist.pdf1(6) == pytest.approx(0, abs=0.1)


def compare_runtimes():
    from tests.conftest import make_histogram

    histogram = make_histogram()
    import time

    start = time.time()
    test_mixed_1(histogram)
    mid = time.time()
    print(f"Total time (1): {mid - start:.2f}s")
    test_mixed_2(histogram)
    print(f"Total time (2): {time.time() - mid:.2f}s")
