from dataclasses import dataclass

import pytest

from ergo import Logistic, LogisticMixture
from ergo.distributions.conditions import HistogramCondition, IntervalCondition


@dataclass
class Uniform:
    min: float = 0
    max: float = 1

    def cdf(self, value):
        lt_min = value < self.min
        gt_max = value > self.max
        in_range = 1 - lt_min * gt_max
        range_val = (value - self.min) / (self.max - self.min)
        return lt_min * 0 + in_range * range_val + gt_max * 1

    def destructure(self):
        return (Uniform, (self.min, self.max))

    @classmethod
    def structure(self, params):
        return Uniform(params[0], params[1])


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
    def normalization_interval_condition_test(p, min, max, scale_min, scale_max):
        condition = IntervalCondition(p=p, min=min, max=max)
        assert (
            condition.normalize(scale_min=scale_min, scale_max=scale_max).denormalize(
                scale_min=scale_min, scale_max=scale_max
            )
            == condition
        )

    # straightforward scenario
    normalization_interval_condition_test(
        p=0.5, min=10, max=100, scale_min=10, scale_max=1000
    )

    # left open
    normalization_interval_condition_test(
        p=0.5, min=None, max=10000, scale_min=10, scale_max=1000
    )

    # right open
    normalization_interval_condition_test(
        p=0.5, min=10, max=None, scale_min=10, scale_max=1000
    )

    # negative values
    normalization_interval_condition_test(
        p=0.5, min=-1000, max=-100, scale_min=-10000, scale_max=-1000
    )

    # p = 1
    normalization_interval_condition_test(
        p=1, min=10, max=100, scale_min=10, scale_max=1000
    )

    # interval bigger than scale
    normalization_interval_condition_test(
        p=1, min=0, max=1000, scale_min=10, scale_max=100
    )

    assert IntervalCondition(p=0.5, min=0, max=5).normalize(
        scale_min=0, scale_max=10
    ) == IntervalCondition(p=0.5, min=0, max=0.5)


def test_normalization_histogram_condition(histogram):
    original = HistogramCondition(histogram["xs"], histogram["densities"])
    normalized_denormalized = original.normalize(10, 1000).denormalize(10, 1000)
    for (density, norm_denorm_density) in zip(
        histogram["densities"], normalized_denormalized.densities
    ):
        assert density == pytest.approx(norm_denorm_density, rel=0.001,)
    for (x, norm_denorm_x) in zip(histogram["xs"], normalized_denormalized.xs):
        assert x == pytest.approx(norm_denorm_x, rel=0.001,)

    # half-assed test that xs and densities are at least
    # getting transformed in the right direction
    normalized = original.normalize(1, 4)
    for idx, (normalized_x, normalized_density) in enumerate(
        zip(normalized.xs, normalized.densities)
    ):
        orig_x = histogram["xs"][idx]
        orig_density = histogram["densities"][idx]
        assert orig_x > normalized_x
        assert orig_density < normalized_density


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
    conditions = [HistogramCondition(histogram["xs"], histogram["densities"])]
    mixture = LogisticMixture.from_conditions(
        conditions, num_components=3, verbose=True
    )
    for (x, density) in zip(histogram["xs"], histogram["densities"]):
        assert mixture.pdf1(x) == pytest.approx(density, abs=0.2)


def test_weights_mixture():
    conditions = [
        IntervalCondition(p=0.4, max=1, weight=0.01),
        IntervalCondition(p=0.5, max=2, weight=100),
        IntervalCondition(p=0.8, max=2.2, weight=0.01),
        IntervalCondition(p=0.9, max=2.3, weight=0.01),
    ]
    dist = LogisticMixture.from_conditions(conditions, num_components=1, verbose=True)
    assert dist.components[0].loc == pytest.approx(2, rel=0.1)


def test_mixed_1(histogram):
    conditions = (
        IntervalCondition(p=0.4, max=1),
        IntervalCondition(p=0.45, max=1.2),
        IntervalCondition(p=0.47, max=1.3),
        IntervalCondition(p=0.5, max=2),
        IntervalCondition(p=0.8, max=2.2),
        IntervalCondition(p=0.9, max=2.3),
        HistogramCondition(histogram["xs"], histogram["densities"]),
    )
    dist = LogisticMixture.from_conditions(conditions, num_components=3, verbose=True)
    assert dist.pdf1(-5) == pytest.approx(0, abs=0.1)
    assert dist.pdf1(6) == pytest.approx(0, abs=0.1)
    my_cache = {}
    my_cache[conditions] = 2
    conditions_2 = (
        IntervalCondition(p=0.4, max=1),
        IntervalCondition(p=0.45, max=1.2),
        IntervalCondition(p=0.47, max=1.3),
        IntervalCondition(p=0.5, max=2),
        IntervalCondition(p=0.8, max=2.2),
        IntervalCondition(p=0.9, max=2.3),
        HistogramCondition(histogram["xs"], histogram["densities"]),
    )
    assert hash(conditions) == hash(conditions_2)
    assert my_cache[conditions_2] == 2


def test_mixed_2(histogram):
    conditions = (
        HistogramCondition(histogram["xs"], histogram["densities"]),
        IntervalCondition(p=0.4, max=1),
        IntervalCondition(p=0.45, max=1.2),
        IntervalCondition(p=0.48, max=1.3),
        IntervalCondition(p=0.5, max=2),
        IntervalCondition(p=0.7, max=2.2),
        IntervalCondition(p=0.9, max=2.3),
    )
    dist = LogisticMixture.from_conditions(conditions, num_components=3, verbose=True)
    assert dist.pdf1(-5) == pytest.approx(0, abs=0.1)
    assert dist.pdf1(6) == pytest.approx(0, abs=0.1)
    my_cache = {}
    my_cache[conditions] = 3
    conditions_2 = (
        HistogramCondition(histogram["xs"], histogram["densities"]),
        IntervalCondition(p=0.4, max=1),
        IntervalCondition(p=0.45, max=1.2),
        IntervalCondition(p=0.48, max=1.3),
        IntervalCondition(p=0.5, max=2),
        IntervalCondition(p=0.7, max=2.2),
        IntervalCondition(p=0.9, max=2.3),
    )
    assert hash(conditions) == hash(conditions_2)
    assert my_cache[conditions_2] == 3


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
