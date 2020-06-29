from dataclasses import dataclass

import jax.numpy as np
import pytest

from ergo.conditions import (
    IntervalCondition,
    MaxEntropyCondition,
    MeanCondition,
    ModeCondition,
    PointDensityCondition,
    SmoothnessCondition,
    VarianceCondition,
)
from ergo.distributions import Logistic, LogisticMixture, PointDensity
from ergo.scale import Scale


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
        return ((Uniform,), (self.min, self.max))

    @classmethod
    def structure(self, params):
        numeric_params = params[1]
        return Uniform(numeric_params[0], numeric_params[1])


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
    def normalization_interval_condition_test(p, min, max, low, high):
        condition = IntervalCondition(p=p, min=min, max=max)
        scale = Scale(low, high)
        assert condition.normalize(scale).denormalize(scale) == condition

    # straightforward scenario
    normalization_interval_condition_test(p=0.5, min=10, max=100, low=10, high=1000)

    # left open
    normalization_interval_condition_test(p=0.5, min=None, max=10000, low=10, high=1000)

    # right open
    normalization_interval_condition_test(p=0.5, min=10, max=None, low=10, high=1000)

    # negative values
    normalization_interval_condition_test(
        p=0.5, min=-1000, max=-100, low=-10000, high=-1000
    )

    # p = 1
    normalization_interval_condition_test(p=1, min=10, max=100, low=10, high=1000)

    # interval bigger than scale
    normalization_interval_condition_test(p=1, min=0, max=1000, low=10, high=100)

    assert IntervalCondition(p=0.5, min=0, max=5).normalize(
        Scale(0, 10)
    ) == IntervalCondition(p=0.5, min=0, max=0.5)


def test_normalization_point_densities_condition(point_densities):
    original = PointDensityCondition(
        point_densities["xs"], point_densities["densities"]
    )
    normalized_denormalized = original.normalize(Scale(10, 1000)).denormalize(
        Scale(10, 1000)
    )
    for (density, norm_denorm_density) in zip(
        point_densities["densities"], normalized_denormalized.densities
    ):
        assert density == pytest.approx(norm_denorm_density, rel=0.001,)
    for (x, norm_denorm_x) in zip(point_densities["xs"], normalized_denormalized.xs):
        assert x == pytest.approx(norm_denorm_x, rel=0.001,)

    # half-assed test that xs and densities are at least
    # getting transformed in the right direction
    normalized = original.normalize(Scale(1, 4))
    for idx, (normalized_x, normalized_density) in enumerate(
        zip(normalized.xs, normalized.densities)
    ):
        orig_x = point_densities["xs"][idx]
        orig_density = point_densities["densities"][idx]
        assert orig_x > normalized_x
        assert orig_density < normalized_density


def test_mixture_from_percentile():
    for value in [0.01, 0.1, 1, 3]:
        conditions = [IntervalCondition(p=0.5, max=value)]
        dist = LogisticMixture.from_conditions(
            conditions, {"num_components": 1}, verbose=True, scale=Scale(0, 3)
        )
        loc = dist.components[0].base_dist.true_loc
        assert loc == pytest.approx(value, rel=0.1), loc


def test_mixture_from_percentiles():
    conditions = [
        IntervalCondition(p=0.1, max=1),
        IntervalCondition(p=0.5, max=2),
        IntervalCondition(p=0.6, max=3),
    ]
    dist = LogisticMixture.from_conditions(
        conditions, {"num_components": 4}, verbose=False, scale=Scale(0, 3)
    )
    for condition in conditions:
        assert dist.cdf(condition.max) == pytest.approx(condition.p, rel=0.1)


def test_percentiles_from_mixture():
    xscale = Scale(-1, 4)
    mixture = LogisticMixture(
        components=[
            Logistic(loc=1, s=0.1, scale=xscale),
            Logistic(loc=2, s=0.1, scale=xscale),
        ],
        probs=[0.5, 0.5],
    )
    conditions = mixture.percentiles(percentiles=[0.1, 0.5, 0.9])
    for condition in conditions:
        if condition.max == 0.5:
            assert condition.p == pytest.approx(1.5, rel=0.01)
    return conditions


@pytest.mark.parametrize(
    "fixed_params",
    [{"num_components": 4}, {"num_components": 4, "floor": 0, "ceiling": 4}],
)
def test_percentile_roundtrip(fixed_params):
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
        conditions, fixed_params, scale=Scale(0, 4), verbose=True,
    )
    recovered_conditions = mixture.percentiles(
        percentiles=[condition.p for condition in conditions]
    )
    for (condition, recovered_condition) in zip(conditions, recovered_conditions):
        assert recovered_condition.max == pytest.approx(condition.max, rel=0.1)


def test_mixture_from_point_densities(point_densities):
    conditions = [
        PointDensityCondition(point_densities["xs"], point_densities["densities"])
    ]

    mixture = LogisticMixture.from_conditions(
        conditions,
        {"num_components": 3},
        Scale(min(point_densities["xs"]), max(point_densities["xs"])),
    )
    for (x, density) in zip(point_densities["xs"], point_densities["densities"]):
        assert mixture.pdf(x) == pytest.approx(density, abs=0.2)


def test_weights_mixture():
    conditions = [
        IntervalCondition(p=0.4, max=1, weight=0.01),
        IntervalCondition(p=0.5, max=2, weight=100),
        IntervalCondition(p=0.8, max=2.2, weight=0.01),
        IntervalCondition(p=0.9, max=2.3, weight=0.01),
    ]
    dist = LogisticMixture.from_conditions(
        conditions, {"num_components": 1}, verbose=True, scale=Scale(0, 3)
    )
    assert dist.components[0].base_dist.true_loc == pytest.approx(2, rel=0.1)


def test_mode_condition():
    base_conditions = [IntervalCondition(p=0.4, max=0.5)]
    base_dist = PointDensity.from_conditions(base_conditions, verbose=True)

    # Most likely condition should increase chance of specified outcome
    outcome_conditions = base_conditions + [ModeCondition(outcome=0.25)]
    outcome_dist = PointDensity.from_conditions(outcome_conditions, verbose=True)
    assert outcome_dist.pdf(0.25) > base_dist.pdf(0.25)

    # Highly weighted most likely condition should make specified outcome most likely
    strong_condition = ModeCondition(outcome=0.25, weight=100000)
    strong_outcome_conditions = base_conditions + [strong_condition]
    strong_outcome_dist = PointDensity.from_conditions(
        strong_outcome_conditions, verbose=True
    )
    assert strong_condition.loss(strong_outcome_dist) == pytest.approx(0, abs=0.001)


def test_mean_condition():
    base_conditions = [MaxEntropyCondition(weight=0.1)]
    base_dist = PointDensity.from_conditions(base_conditions, verbose=True)
    base_mean = base_dist.mean()

    # Mean condition should move mean closer to specified mean
    mean_conditions = base_conditions + [MeanCondition(mean=0.25, weight=1)]
    mean_dist = PointDensity.from_conditions(mean_conditions, verbose=True)
    assert abs(mean_dist.mean() - 0.25) < abs(base_mean - 0.25)

    # Highly weighted mean condition should make mean very close to specified mean
    strong_condition = MeanCondition(mean=0.25, weight=100000)
    strong_mean_conditions = base_conditions + [strong_condition]
    strong_mean_dist = PointDensity.from_conditions(
        strong_mean_conditions, verbose=True
    )
    assert strong_mean_dist.mean() == pytest.approx(0.25, rel=0.01)


def test_variance_condition():
    base_conditions = [
        MaxEntropyCondition(weight=0.1),
        SmoothnessCondition(),
        IntervalCondition(p=0.95, min=0.3, max=0.7),
    ]
    base_dist = PointDensity.from_conditions(base_conditions, verbose=True)
    base_variance = base_dist.variance()
    increased_variance = base_variance + 0.01

    # Increase in variance should decrease peak
    var_condition = VarianceCondition(variance=increased_variance, weight=1)
    var_conditions = base_conditions + [var_condition]
    var_dist = PointDensity.from_conditions(var_conditions, verbose=True)
    assert np.max(var_dist.normed_densities) < np.max(base_dist.normed_densities)

    # Highly weighted variance condition should make var very close to specified var
    strong_condition = VarianceCondition(variance=increased_variance, weight=100000)
    strong_var_conditions = base_conditions + [strong_condition]
    strong_var_dist = PointDensity.from_conditions(strong_var_conditions, verbose=True)
    assert strong_var_dist.variance() == pytest.approx(
        float(increased_variance), abs=0.001
    )


def test_mixed_1(point_densities):
    conditions = (
        IntervalCondition(p=0.4, max=1),
        IntervalCondition(p=0.45, max=1.2),
        IntervalCondition(p=0.47, max=1.3),
        IntervalCondition(p=0.5, max=2),
        IntervalCondition(p=0.8, max=2.2),
        IntervalCondition(p=0.9, max=2.3),
        PointDensityCondition(point_densities["xs"], point_densities["densities"]),
    )
    dist = LogisticMixture.from_conditions(
        conditions, {"num_components": 3}, verbose=True
    )
    assert dist.pdf(-5) == pytest.approx(0, abs=0.1)
    assert dist.pdf(6) == pytest.approx(0, abs=0.1)
    my_cache = {}
    my_cache[conditions] = 2
    conditions_2 = (
        IntervalCondition(p=0.4, max=1),
        IntervalCondition(p=0.45, max=1.2),
        IntervalCondition(p=0.47, max=1.3),
        IntervalCondition(p=0.5, max=2),
        IntervalCondition(p=0.8, max=2.2),
        IntervalCondition(p=0.9, max=2.3),
        PointDensityCondition(point_densities["xs"], point_densities["densities"]),
    )
    assert hash(conditions) == hash(conditions_2)
    assert my_cache[conditions_2] == 2


def test_mixed_2(point_densities):
    conditions = (
        PointDensityCondition(point_densities["xs"], point_densities["densities"]),
        IntervalCondition(p=0.4, max=1),
        IntervalCondition(p=0.45, max=1.2),
        IntervalCondition(p=0.48, max=1.3),
        IntervalCondition(p=0.5, max=2),
        IntervalCondition(p=0.7, max=2.2),
        IntervalCondition(p=0.9, max=2.3),
    )
    dist = LogisticMixture.from_conditions(
        conditions, {"num_components": 3}, verbose=True
    )
    assert dist.pdf(-5) == pytest.approx(0, abs=0.1)
    assert dist.pdf(6) == pytest.approx(0, abs=0.1)
    my_cache = {}
    my_cache[conditions] = 3
    conditions_2 = (
        PointDensityCondition(point_densities["xs"], point_densities["densities"]),
        IntervalCondition(p=0.4, max=1),
        IntervalCondition(p=0.45, max=1.2),
        IntervalCondition(p=0.48, max=1.3),
        IntervalCondition(p=0.5, max=2),
        IntervalCondition(p=0.7, max=2.2),
        IntervalCondition(p=0.9, max=2.3),
    )
    assert hash(conditions) == hash(conditions_2)
    assert my_cache[conditions_2] == 3


def test_point_densities_fit(point_densities):
    condition = PointDensityCondition(
        point_densities["xs"], point_densities["densities"]
    )
    conditions = (condition,)

    dist = PointDensity.from_conditions(
        conditions,
        scale=Scale(min(point_densities["xs"]), max(point_densities["xs"])),
        verbose=True,
    )

    for (original_x, original_density) in zip(
        point_densities["xs"], point_densities["densities"]
    ):
        assert dist.pdf(original_x) == pytest.approx(original_density, abs=0.05)


@pytest.mark.xfail(reason="Will hopefully be solved by switching to PointDensity")
def test_point_density_fit_regression_p_in_range():
    """
    Regression test for a bug where:
    1. < 100% of p is in the entire range, for a closed-bound question
    2. the p is smashed up against the edges of the range
    rather than distributed evenly over the whole range

    e.g. see https://elicit.ought.org/builder/Mib4yBPDE

    For more on the bug, see
    https://docs.google.com/document/d/1CFklTKtbKzXi6-lRaEsX4ZiY3Yzpbfdg7i2j1NvKP34/edit#heading=h.lypz52bknpyq
    """
    pointdensity_dist = PointDensity.from_conditions(
        conditions=[IntervalCondition(min=0, max=1, p=0.5)]
    )

    assert pointdensity_dist.cdf(1) == 1


@pytest.mark.xfail(reason="Will hopefully be solved by switching to PointDensity")
def test_fit_hist_regression_1():
    """
    Regression test for bug: "This custom question has a weird histogram - why?"

    see https://elicit.ought.org/builder/gflpsSBAb

    for more on the bug, see
    https://docs.google.com/document/d/1CFklTKtbKzXi6-lRaEsX4ZiY3Yzpbfdg7i2j1NvKP34/edit#heading=h.ph1huakxn33f
    """
    conditions = [
        IntervalCondition(p=0.25, max=2.0),
        IntervalCondition(p=0.75, max=4.0),
        IntervalCondition(p=0.9, max=6.0),
        MaxEntropyCondition(weight=0.1),
    ]

    pointdensity_dist = PointDensity.from_conditions(
        conditions, scale=Scale(low=0, high=52)
    )

    assert pointdensity_dist.cdf(2) == pytest.approx(0.25, abs=0.05)
    assert pointdensity_dist.ppf(0.9) == pytest.approx(6, abs=1)


def compare_runtimes(point_densities):
    import time

    start = time.time()
    test_mixed_1(point_densities)
    mid = time.time()
    print(f"Total time (1): {mid - start:.2f}s")
    test_mixed_2(point_densities)
    print(f"Total time (2): {time.time() - mid:.2f}s")
