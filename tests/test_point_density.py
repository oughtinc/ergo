import numpy as np
import pytest
from scipy.stats import logistic

from ergo.conditions import (
    CrossEntropyCondition,
    IntervalCondition,
    MaxEntropyCondition,
    PointDensityCondition,
)
from ergo.distributions.point_density import PointDensity
from ergo.scale import LogScale, Scale
from tests.conftest import scales_to_test


@pytest.mark.parametrize(
    "scale",
    [Scale(0, 5), Scale(-1, 6), Scale(-3, 10), LogScale(0.01, 5, 500)],
    #    "scale", [LogScale(0.01, 5, 500)],
)
@pytest.mark.parametrize(
    "dist_source",
    ["direct", "from_pairs", "structured", "denormalized", "from_conditions"],
)
def test_point_density(scale, dist_source):
    rv = logistic(loc=2.5, scale=0.15)
    xs = np.linspace(
        0.01, 5, 100
    )  
    orig_densities = rv.pdf(xs)
    orig_cdfs = rv.cdf(xs)

    direct_dist = PointDensity(xs, orig_densities, scale)

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
        # cond = CrossEntropyCondition(p_dist=direct_dist)
        xs, densities = direct_dist.to_lists()
        cond = PointDensityCondition(xs, densities)
        dist = PointDensity.from_conditions(
            [cond], fixed_params={"xs": xs}, scale=scale
        )

    # PDF
    dist_densities = np.array([float(dist.pdf(x)) for x in xs])
    #print(f"scale: {scale} dist_source: {dist_source}")
    #print(f"max pdf diff: {np.max(np.abs(orig_densities-dist_densities))}")
    assert dist_densities == pytest.approx(orig_densities, abs=.01)

    # CDF
    dist_cdfs = np.array([float(dist.cdf(x)) for x in xs])
    #print(f"max cdf diff: {np.max(np.abs(orig_cdfs-dist_cdfs))}")
    assert dist_cdfs == pytest.approx(orig_cdfs, abs=.05)

    # PPF has low resolution at the low end (because distribution is
    # flat) and at high end (because distribution is flat and log
    # scale is coarse there)
    dist_ppfs = np.array([float(dist.ppf(c)) for c in orig_cdfs[30:60]])
    #print(f"max ppf diff: {np.max(np.abs(xs[30:60]-dist_ppfs))}")
    #import pdb; pdb.set_trace()
    assert dist_ppfs == pytest.approx(xs[30:60], abs=.1)


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
def test_hist_ppf(scale: Scale):
    uniform_dist = PointDensity.from_conditions([], scale=scale)

    # Ends of scale; second is approx since implemented as start of last bin
    assert uniform_dist.ppf(0) == scale.low
    assert uniform_dist.ppf(1) == pytest.approx(scale.high, rel=0.1)
