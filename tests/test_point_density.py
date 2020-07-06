import numpy as np
import pytest
from scipy.stats import logistic

from ergo.conditions import (
    CrossEntropyCondition,
    IntervalCondition,
    MaxEntropyCondition,
)
from ergo.distributions.point_density import PointDensity
from ergo.scale import LogScale, Scale
from tests.conftest import scales_to_test


@pytest.mark.parametrize(
    "scale", [Scale(0, 5), Scale(-1, 6), Scale(-3, 10), LogScale(0.01, 5, 500)],
)
@pytest.mark.parametrize(
    "dist_source",
    ["direct", "from_pairs", "structured", "denormalized", "from_conditions"],
)
def test_point_density(scale, dist_source):
    NUM_POINTS = 200

    rv = logistic(loc=2.5, scale=0.15)
    xs_normed = np.linspace(0, 1, NUM_POINTS + 1)
    xs_grid_normed = (xs_normed[:-1] + xs_normed[1:]) / 2
    xs = scale.denormalize_points(xs_grid_normed)

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
    elif dist_source == "to_pairs":
        pairs = direct_dist.to_pairs()
        dist = PointDensity.from_pairs(pairs, scale)
    elif dist_source == "structured":
        dist = PointDensity.structure(direct_dist.destructure())
    elif dist_source == "denormalized":
        dist = direct_dist.normalize().denormalize(scale)
    elif dist_source == "from_conditions":
        cond = CrossEntropyCondition(p_dist=direct_dist)
        dist = PointDensity.from_conditions([cond], scale=scale)

    print(f"source: {dist_source} scale: {scale}")
    # PDF
    dist_densities = np.array([float(dist.pdf(x)) for x in xs])
    print(f"max pdf diff: {np.max(np.abs(dist_densities-orig_densities))}")
    """
    idx = np.argmax(np.abs(dist_densities-orig_densities))
    print(f'idx {idx} dist {dist_densities[idx]} orig {orig_densities[idx]}')
    print(f'pdf diffs: {dist_densities-orig_densities}')
    import pdb; pdb.set_trace()
    """
    assert dist_densities == pytest.approx(orig_densities, abs=0.01)

    # CDF
    dist_cdfs = np.array([float(dist.cdf(x)) for x in xs])
    print(f"max cdf diff: {np.max(np.abs(dist_cdfs-orig_cdfs))}")
    """
    idx = np.argmax(np.abs(dist_cdfs - orig_cdfs))
    print(f'idx {idx} dist {dist_cdfs[idx]} orig {orig_cdfs[idx]}')
    print(f'cdf diffs: {dist_cdfs-orig_cdfs}')
    """
    assert dist_cdfs == pytest.approx(orig_cdfs, abs=0.05)

    # PPF has low resolution at the low end (because distribution is
    # flat) and at high end (because distribution is flat and log
    # scale is coarse there)
    MIN_CHECK_DENSITY = 1e-3
    check_idxs = [i for i in range(NUM_POINTS) if orig_densities[i] > MIN_CHECK_DENSITY]
    dist_ppfs = np.array([float(dist.ppf(c)) for c in orig_cdfs[check_idxs]])
    print(f"max ppf diff: {np.max(np.abs(dist_ppfs-xs[check_idxs]))}")
    assert dist_ppfs == pytest.approx(xs[check_idxs], abs=0.1)


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
        dist = PointDensity.from_conditions(conditions)
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
    NUM_POINTS = 11

    """
    HANDPICKED_PAIRS = [
        {"x": 0, "density": 5 / 3},
        {"x": 0.1, "density": 5 / 3},
        {"x": 0.2, "density": 5 / 3},
        {"x": 0.3, "density": 5 / 3},
        {"x": 0.4, "density": 25 / 39},
        {"x": 0.5, "density": 25 / 39},
        {"x": 0.6, "density": 25 / 39},
        {"x": 0.7, "density": 25 / 39},
        {"x": 0.8, "density": 25 / 39},
        {"x": 0.9, "density": 25 / 39},
        {"x": 1, "density": 25 / 39},
    ]

    handpicked_dist = PointDensity.from_pairs(HANDPICKED_PAIRS, scale, normalized=True)
    """

    conditions = [
        IntervalCondition(p=0.5, max=scale.denormalize_point(0.3)),
        MaxEntropyCondition(weight=0.01),
    ]

    fitted_dist = PointDensity.from_conditions(
        conditions, scale=scale, num_points=NUM_POINTS,
    )

    def evaluate_dist(dist):
        loss = 0
        for condition in conditions:
            print(condition._describe_fit(dist))
            loss += condition.loss(dist)
        print(f"nd: {dist.normed_densities} total loss: {loss}")

    """
    print("evaluating handpicked dist")
    evaluate_dist(handpicked_dist)
    """
    print("evaluating fitted dist")
    evaluate_dist(fitted_dist)
    print(
        f"fitted dist midpoints: {(fitted_dist.normed_densities[1:] + fitted_dist.normed_densities[:-1]) / 2.0}"
    )
    # We expect at most 3 different densities: one for inside the interval, one for outside,
    # and one between.
    assert np.unique(fitted_dist.normed_densities).size <= 3
