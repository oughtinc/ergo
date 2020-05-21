from dataclasses import dataclass

import pytest

from ergo import Logistic, LogisticMixture
from ergo.distributions.conditions import HistogramCondition, IntervalCondition


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


def test_interval_condition():
    dist = Uniform(min=-1, max=1)

    assert IntervalCondition(p=0.5, min=0, max=1).loss(dist) == 0
    assert IntervalCondition(p=0.25, min=0, max=1).loss(dist) == 0.25 ** 2
    assert IntervalCondition(p=1, max=0).loss(dist) == 0.5 ** 2
    assert IntervalCondition(p=1).loss(dist) == 0
    assert IntervalCondition(p=0, min=-1, max=1).loss(dist) == 1
    assert IntervalCondition(p=0, min=-1, max=1, weight=10).loss(dist) == 10

    assert (
        IntervalCondition(p=0.25, min=0, max=1).describe_fit(dist)["loss"] == 0.25 ** 2
    )
    assert (
        IntervalCondition(p=0, min=-1, max=0).describe_fit(dist)["p_in_interval"] == 0.5
    )
    assert (
        IntervalCondition(p=1, min=-1, max=0).describe_fit(dist)["p_in_interval"] == 0.5
    )


def test_normalization_interval_condition():
    condition = IntervalCondition(p=0.5, min=10, max=100)
    assert condition.get_normalized(10, 1000).get_denormalized(10, 1000) == condition


def test_normalization_histogram_condition(histogram):
    original = HistogramCondition(histogram)
    normalized_denormalized = original.get_normalized(10, 1000).get_denormalized(
        10, 1000
    )
    for entry in normalized_denormalized.histogram:
        assert entry["x"] == pytest.approx(
            [
                original_entry["x"]
                for original_entry in original.histogram
                if original_entry["density"] == entry["density"]
            ][0],
            rel=0.001,
        )


def test_mixture_from_percentile():
    for value in [0.01, 0.1, 1, 3]:
        conditions = [IntervalCondition(p=0.5, max=value)]
        dist = LogisticMixture.from_conditions(
            conditions, num_components=1, verbose=True
        )
        loc = dist.components[0].loc
        assert loc == pytest.approx(value, rel=0.1), loc


def test_mixture_from_percentiles():
    conditions = [
        IntervalCondition(p=0.1, max=1),
        IntervalCondition(p=0.5, max=2),
        IntervalCondition(p=0.6, max=3),
    ]
    dist = LogisticMixture.from_conditions(conditions, num_components=3, verbose=True)
    for condition in conditions:
        assert dist.cdf(condition.max) == pytest.approx(condition.p, rel=0.1)


def test_percentiles_from_mixture():
    mixture = LogisticMixture(
        components=[Logistic(loc=1, scale=0.1), Logistic(loc=2, scale=0.1)],
        probs=[0.5, 0.5],
    )
    conditions = mixture.percentiles(percentiles=[0.1, 0.5, 0.9])
    for condition in conditions:
        if condition.max == 0.5:
            assert condition.p == pytest.approx(1.5, rel=0.01)
    return conditions


def test_percentile_roundtrip():
    conditions = [
        IntervalCondition(p=0.01, max=0.61081324517545),
        IntervalCondition(p=0.1, max=0.8613634657212543),
        IntervalCondition(p=0.25, max=1),
        IntervalCondition(p=0.5, max=1.5),
        IntervalCondition(p=0.75, max=2),
        IntervalCondition(p=0.9, max=2.1386364698410034),
        IntervalCondition(p=0.99, max=2.3891870975494385),
    ]
    mixture = LogisticMixture.from_conditions(
        conditions, num_components=3, verbose=True
    )
    recovered_conditions = mixture.percentiles(
        percentiles=[condition.p for condition in conditions]
    )
    for (condition, recovered_condition) in zip(conditions, recovered_conditions):
        assert recovered_condition.max == pytest.approx(condition.max, rel=0.1)


def test_mixture_from_histogram(histogram):
    conditions = [HistogramCondition(histogram)]
    mixture = LogisticMixture.from_conditions(
        conditions, num_components=3, verbose=True
    )
    for entry in histogram:
        assert mixture.pdf1(entry["x"]) == pytest.approx(entry["density"], abs=0.2)


def test_weights_mixture():
    conditions = [
        IntervalCondition(p=0.4, max=1, weight=0.01),
        IntervalCondition(p=0.5, max=2, weight=100),
        IntervalCondition(p=0.8, max=2.2, weight=0.01),
        IntervalCondition(p=0.9, max=2.3, weight=0.01),
    ]
    dist = LogisticMixture.from_conditions(conditions, num_components=1, verbose=True)
    assert dist.components[0].loc == pytest.approx(2, rel=0.1)


def test_histogram_condition(histogram, normalized_logistic_mixture):
    condition = HistogramCondition(histogram)
    vectorized_loss = float(condition.loss(normalized_logistic_mixture))
    loop_loss = float(condition._loss_loop(normalized_logistic_mixture))
    assert vectorized_loss == pytest.approx(loop_loss, rel=0.0001)


def test_mixed_1(histogram):
    conditions = [
        IntervalCondition(p=0.4, max=1),
        IntervalCondition(p=0.45, max=1.2),
        IntervalCondition(p=0.47, max=1.3),
        IntervalCondition(p=0.5, max=2),
        IntervalCondition(p=0.8, max=2.2),
        IntervalCondition(p=0.9, max=2.3),
        HistogramCondition(histogram),
    ]
    dist = LogisticMixture.from_conditions(conditions, num_components=3, verbose=True)
    assert dist.pdf1(-5) == pytest.approx(0, abs=0.1)
    assert dist.pdf1(6) == pytest.approx(0, abs=0.1)


def test_mixed_2(histogram):
    conditions = [
        HistogramCondition(histogram),
        IntervalCondition(p=0.4, max=1),
        IntervalCondition(p=0.45, max=1.2),
        IntervalCondition(p=0.48, max=1.3),
        IntervalCondition(p=0.5, max=2),
        IntervalCondition(p=0.7, max=2.2),
        IntervalCondition(p=0.9, max=2.3),
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
