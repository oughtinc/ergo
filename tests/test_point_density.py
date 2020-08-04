import numpy as np
import pytest
from scipy.stats import logistic, norm

from ergo.conditions import (
    CrossEntropyCondition,
    IntervalCondition,
    MaxEntropyCondition,
)
import ergo.distributions.constants as constants
from ergo.distributions.point_density import PointDensity
from ergo.scale import LogScale, Scale
from tests.conftest import scales_to_test


@pytest.mark.parametrize(
    "scale", [Scale(0, 5), Scale(-1, 6), Scale(-3, 10), LogScale(0.01, 5, 500)],
)
@pytest.mark.parametrize(
    "dist_source",
    [
        "direct",
        "from_pairs",
        "structured",
        "denormalized",
        "from_conditions",
        "to_arrays",
        "to_arrays/2",
    ],
)
def test_point_density(scale, dist_source):
    scale_mid = scale.low + scale.width / 2
    rv = logistic(loc=scale_mid, scale=scale.width / 30)
    xs = scale.denormalize_points(constants.target_xs)

    orig_densities = rv.pdf(xs)
    orig_cdfs = rv.cdf(xs)

    orig_pairs = [
        {"x": x, "density": density} for (x, density) in zip(xs, orig_densities)
    ]
    direct_dist = PointDensity.from_pairs(orig_pairs, scale)

    if dist_source == "direct":
        dist = direct_dist
    elif dist_source == "from_pairs":
        orig_pairs = [
            {"x": x, "density": density} for (x, density) in zip(xs, orig_densities)
        ]
        dist = PointDensity.from_pairs(orig_pairs, scale)
    elif dist_source == "to_arrays":
        _xs, _density = direct_dist.to_arrays()
        pairs = [{"x": x, "density": d} for x, d in zip(_xs, _density)]
        dist = PointDensity.from_pairs(pairs, scale)
    elif dist_source == "to_arrays/2":
        _xs, _density = direct_dist.to_arrays(
            num_xs=int(constants.point_density_default_num_points / 2),
            add_endpoints=True,
        )
        pairs = [{"x": x, "density": d} for x, d in zip(_xs, _density)]
        dist = PointDensity.from_pairs(pairs, scale)
    elif dist_source == "structured":
        dist = PointDensity.structure(direct_dist.destructure())
    elif dist_source == "denormalized":
        dist = direct_dist.normalize().denormalize(scale)
    elif dist_source == "from_conditions":
        cond = CrossEntropyCondition(p_dist=direct_dist)
        dist = PointDensity.from_conditions([cond], scale=scale)

    # PDF
    dist_densities = np.array([float(dist.pdf(x)) for x in xs])
    if dist_source == "to_arrays/2":
        assert dist_densities == pytest.approx(orig_densities, abs=0.05)
    else:
        assert dist_densities == pytest.approx(orig_densities, abs=0.01)

    # CDF
    dist_cdfs = np.array([float(dist.cdf(x)) for x in xs])
    assert dist_cdfs == pytest.approx(orig_cdfs, abs=0.05)
    MIN_CHECK_DENSITY = 1e-3
    check_idxs = [
        i
        for i in range(constants.point_density_default_num_points)
        if orig_densities[i] > MIN_CHECK_DENSITY
    ]
    dist_ppfs = np.array([float(dist.ppf(c)) for c in orig_cdfs[check_idxs]])
    assert dist_ppfs == pytest.approx(xs[check_idxs], rel=0.1)


def test_density_frompairs():
    pairs = [
        {"x": 0, "density": 1},
        {"x": 0.2, "density": 1},
        {"x": 0.4, "density": 1},
        {"x": 0.6, "density": 1},
        {"x": 1, "density": 1},
    ]
    dist = PointDensity.from_pairs(pairs, scale=Scale(0, 1))
    for condition in dist.percentiles():
        assert condition.max == pytest.approx(condition.p, abs=0.01)


def test_density_percentile():
    for value in [0.01, 0.1, 0.5, 0.9]:
        conditions = [IntervalCondition(p=0.5, max=value)]
        dist = PointDensity.from_conditions(conditions, scale=Scale(0, 1))
        assert dist.ppf(0.5) == pytest.approx(value, abs=0.1)


@pytest.mark.parametrize("scale", scales_to_test)
def test_density_pdf(scale: Scale):
    uniform_dist = PointDensity.from_conditions([MaxEntropyCondition()], scale=scale)

    assert uniform_dist.pdf(scale.denormalize_point(0.5)) != 0
    assert uniform_dist.pdf(scale.denormalize_point(1.5)) == 0


@pytest.mark.parametrize("scale", scales_to_test)
def test_density_cdf(scale: Scale):
    uniform_dist = PointDensity.from_conditions([MaxEntropyCondition()], scale=scale)

    # Off of scale
    assert uniform_dist.cdf(scale.denormalize_point(-0.5)) == 0
    assert uniform_dist.cdf(scale.denormalize_point(1.5)) == 1

    # Edges of scale
    assert uniform_dist.cdf(scale.denormalize_point(0.005)) < uniform_dist.cdf(
        scale.denormalize_point(0.015)
    )
    assert uniform_dist.cdf(scale.denormalize_point(0.985)) < uniform_dist.cdf(
        scale.denormalize_point(0.995)
    )

    # Denormalized onto a different scale
    denormalized_dist = uniform_dist.denormalize(Scale(0, 2))
    assert denormalized_dist.cdf(1) == pytest.approx(0.5, abs=0.01)
    assert denormalized_dist.cdf(1.5) != 0
    assert denormalized_dist.cdf(2.5) == 1


@pytest.mark.parametrize("scale", scales_to_test)
def test_point_density_ppf(scale: Scale):
    uniform_dist = PointDensity.from_conditions([], scale=scale)

    # Ends of scale; second is approx since implemented as start of last bin
    assert uniform_dist.ppf(0) == scale.low
    assert uniform_dist.ppf(1) == pytest.approx(scale.high, rel=0.1)


@pytest.mark.parametrize("scale", scales_to_test)
def test_interval_plus_entropy(scale: Scale):
    conditions = [
        IntervalCondition(p=0.5, max=scale.denormalize_point(0.3)),
        MaxEntropyCondition(weight=0.01),
    ]

    fitted_dist = PointDensity.from_conditions(conditions, scale=scale,)

    # We expect at most 3 different densities: one for inside the interval, one for outside,
    # and one between.
    assert np.unique(fitted_dist.normed_densities).size <= 3


def test_add_endpoints():
    xs = [0.25, 0.5, 0.75]

    standard_densities = [0.25, 0.5, 0.75]
    expected_densities = np.array([0, 0.25, 0.5, 0.75, 1])

    _, densities = PointDensity.add_endpoints(xs, standard_densities, scale=Scale(0, 1))
    assert densities == pytest.approx(expected_densities, abs=1e-5)

    to_clamp_densities = [0.1, 0.5, 0.1]
    expected_densities = np.array([0, 0.1, 0.5, 0.1, 0])

    _, densities = PointDensity.add_endpoints(xs, to_clamp_densities, scale=Scale(0, 1))
    assert densities == pytest.approx(expected_densities, abs=1e-5)


@pytest.mark.parametrize("scale", scales_to_test)
def test_mean(scale: Scale):
    true_mean = scale.low + scale.width / 2
    rv = norm(loc=true_mean, scale=scale.width / 10)
    xs = constants.target_xs
    pairs = [{"x": x, "density": rv.pdf(x)} for x in scale.denormalize_points(xs)]
    pd_norm = PointDensity.from_pairs(pairs, scale)
    calculated_mean = float(pd_norm.mean())
    assert true_mean == pytest.approx(calculated_mean, rel=1e-3, abs=1e-3)


@pytest.mark.parametrize("scale", scales_to_test)
def test_variance(scale: Scale):
    true_mean = scale.low + scale.width / 2
    true_std = scale.width / 10
    true_variance = true_std ** 2
    rv = norm(loc=true_mean, scale=true_std)
    xs = constants.target_xs
    pairs = [{"x": x, "density": rv.pdf(x)} for x in scale.denormalize_points(xs)]
    pd_norm = PointDensity.from_pairs(pairs, scale)
    calculated_variance = float(pd_norm.variance())
    assert true_variance == pytest.approx(calculated_variance, rel=1e-3, abs=1e-3)
