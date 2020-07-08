import numpy as np
import pytest
import scipy.stats

from ergo.scale import LogScale, Scale, TimeScale, scale_factory
from tests.conftest import scales_to_test


def test_serialization():
    assert hash(Scale(0, 100)) == hash(Scale(0, 100))
    assert hash(Scale(0, 100)) != hash(Scale(100, 200))

    assert hash(LogScale(0, 100, 10)) == hash(LogScale(0, 100, 10))
    assert hash(LogScale(0, 100, 10)) != hash(LogScale(0, 100, 100))

    assert hash(TimeScale(946684800, 1592914415)) == hash(
        TimeScale(946684800, 1592914415)
    )
    assert hash(TimeScale(631152000, 1592914415)) != hash(
        TimeScale(946684800, 1592914415)
    )

    assert (
        hash(LogScale(0, 100, 1))
        != hash(Scale(0, 100))
        != hash(TimeScale(631152000, 946684800))
    )


def test_export_import():
    log_scale = LogScale(low=-1, high=1, log_base=2)
    log_scale_export = log_scale.export()
    assert log_scale_export["width"] == 2
    assert log_scale_export["class"] == "LogScale"

    assert (scale_factory(log_scale.export())) == log_scale

    linear_scale = Scale(low=1, high=10000)
    assert (scale_factory(linear_scale.export())) == linear_scale

    linear_date_scale = TimeScale(low=631152000, high=946684800)
    assert (scale_factory(linear_date_scale.export())) == linear_date_scale


@pytest.mark.look
@pytest.mark.parametrize("scale", scales_to_test)
def test_density_norm_denorm_roundtrip(scale: Scale):
    rv = scipy.stats.logistic(loc=0.5, scale=0.15)
    normed_xs = np.linspace(0.01, 1, 201)
    normed_densities_truth_set = rv.pdf(normed_xs)
    xs = scale.denormalize_points(normed_xs)

    denormed_densities = scale.denormalize_densities(xs, normed_densities_truth_set)
    normed_densities = scale.normalize_densities(normed_xs, denormed_densities)

    assert np.allclose(normed_densities_truth_set, normed_densities)  # type: ignore
