import jax.numpy as np
import numpy as onp
import pytest
import scipy

from ergo import Logistic, LogisticMixture, Truncate
from ergo.conditions import PointDensityCondition
from ergo.scale import LogScale, Scale
from ergo.utils import trapz
from tests.conftest import scales_to_test


@pytest.mark.parametrize("xscale", scales_to_test)
def test_cdf(xscale: Scale):
    scipydist_normed = scipy.stats.logistic(0.5, 0.05)
    true_loc = xscale.denormalize_point(0.5)
    true_s = 0.05 * xscale.width

    ergodist = Logistic(loc=true_loc, s=true_s, scale=xscale)

    for x in np.linspace(0, 1, 10):
        assert scipydist_normed.cdf(x) == pytest.approx(
            float(ergodist.cdf(xscale.denormalize_point(x))), rel=1e-3
        )

    # TODO: consider a better approach for log scale
    if isinstance(xscale, LogScale):
        for x in np.linspace(xscale.low, xscale.high, 10):
            assert scipydist_normed.cdf(xscale.normalize_point(x)) == pytest.approx(
                float(ergodist.cdf(x)), rel=1e-3
            )
    else:
        scipydist_true = scipy.stats.logistic(true_loc, true_s)
        for x in np.linspace(xscale.low, xscale.high, 10):
            assert scipydist_true.cdf(x) == pytest.approx(
                float(ergodist.cdf(x)), rel=1e-3
            )


# TODO test truncated Logistic better in this file


@pytest.mark.parametrize("xscale", scales_to_test)
def test_truncated_ppf(xscale: Scale):
    normed_test_loc = 0.5
    normed_test_s = 0.1
    test_loc = xscale.denormalize_point(normed_test_loc)
    test_s = normed_test_s * xscale.width

    normed_baseline_dist = scipy.stats.logistic(normed_test_loc, normed_test_s)

    def ppf_through_cdf(dist, q):
        return scipy.optimize.bisect(
            lambda x: dist.cdf(x) - q, dist.ppf(0.0001), dist.ppf(0.9999), maxiter=1000
        )

    # No bounds
    dist_w_no_bounds = Truncate(Logistic(loc=test_loc, s=test_s, scale=xscale))

    for x in np.linspace(0.01, 0.99, 8):
        assert dist_w_no_bounds.ppf(x) == pytest.approx(
            float(xscale.denormalize_point(normed_baseline_dist.ppf(x))), rel=0.001
        )

    # Floor
    dist_w_floor = Truncate(
        Logistic(loc=test_loc, s=test_s, scale=xscale),
        floor=xscale.denormalize_point(0.5),
    )

    mix_w_floor = LogisticMixture(
        components=[
            Truncate(  # type: ignore
                Logistic(test_loc, s=test_s, scale=xscale),
                floor=xscale.denormalize_point(0.5),
            )
        ],
        probs=[1.0],
    )

    for x in np.linspace(0.01, 0.99, 8):
        assert dist_w_floor.ppf(x) == pytest.approx(
            float(mix_w_floor.ppf(x)), rel=0.001
        )
        assert dist_w_floor.ppf(x) == pytest.approx(
            float(ppf_through_cdf(dist_w_floor, x)), rel=0.001
        )

    # Ceiling
    dist_w_ceiling = Truncate(
        Logistic(loc=test_loc, s=test_s, scale=xscale),
        ceiling=xscale.denormalize_point(0.8),
    )

    mix_w_ceiling = LogisticMixture(
        components=[
            Truncate(  # type: ignore
                Logistic(test_loc, s=test_s, scale=xscale),
                ceiling=xscale.denormalize_point(0.8),
            )
        ],
        probs=[1.0],
    )

    for x in np.linspace(0.01, 0.99, 8):
        assert dist_w_ceiling.ppf(x) == pytest.approx(
            float(mix_w_ceiling.ppf(x)), rel=0.001
        )
        assert dist_w_ceiling.ppf(x) == pytest.approx(
            float(ppf_through_cdf(dist_w_ceiling, x)), rel=0.001
        )

    # Floor and Ceiling

    dist_w_floor_and_ceiling = Truncate(
        Logistic(loc=test_loc, s=test_s, scale=xscale),
        floor=xscale.denormalize_point(0.2),
        ceiling=xscale.denormalize_point(0.8),
    )

    mix_w_floor_and_ceiling = LogisticMixture(
        components=[
            Truncate(  # type: ignore
                Logistic(test_loc, s=test_s, scale=xscale),
                floor=xscale.denormalize_point(0.2),
                ceiling=xscale.denormalize_point(0.8),
            )
        ],
        probs=[1.0],
    )

    for x in np.linspace(0.01, 0.99, 8):
        assert dist_w_floor_and_ceiling.ppf(x) == pytest.approx(
            float(mix_w_floor_and_ceiling.ppf(x)), rel=0.001
        )
        assert dist_w_floor_and_ceiling.ppf(x) == pytest.approx(
            float(ppf_through_cdf(dist_w_floor_and_ceiling, x)), rel=0.001
        )


@pytest.mark.look
@pytest.mark.parametrize("xscale", scales_to_test)
def test_pdf(xscale: Scale):
    normed_test_loc = 0.5
    normed_test_s = 0.1
    test_loc = xscale.denormalize_point(normed_test_loc)
    test_s = normed_test_s * xscale.width

    ergoLogisticMixture = LogisticMixture(
        components=[
            Logistic(
                loc=xscale.denormalize_point(0.2), s=0.5 * xscale.width, scale=xscale,
            ),
            Logistic(loc=test_loc, s=test_s, scale=xscale),
        ],
        probs=[1.8629593e-29, 1.0],
    )
    ergoLogistic = Logistic(loc=test_loc, s=test_s, scale=xscale)

    ## Make sure it integrates to 1
    _xs = xscale.denormalize_points(np.linspace(0, 1, 100))
    densities_logistic = np.array([float(ergoLogistic.pdf(x)) for x in _xs])
    densities_mixture = np.array([float(ergoLogisticMixture.pdf(x)) for x in _xs])
    auc_logistic = float(trapz(densities_logistic, x=_xs))
    auc_mixture = float(trapz(densities_mixture, x=_xs))
    assert (
        1
        == pytest.approx(auc_logistic, abs=0.03)
        == pytest.approx(auc_mixture, abs=0.03)
    )

    if not isinstance(xscale, LogScale):
        scipydist = scipy.stats.logistic(test_loc, test_s)

        for x in np.linspace(
            xscale.denormalize_point(0), xscale.denormalize_point(1), 10
        ):
            assert (
                scipydist.pdf(x)
                == pytest.approx(float(ergoLogistic.pdf(x)), rel=1e-3)
                == pytest.approx(float(ergoLogisticMixture.pdf(x)), rel=1e-3)
            )


@pytest.mark.xfail(
    reason="We need to devise way of testing true pdf values for our 'log logistic'"
)
@pytest.mark.parametrize("xscale", [scales_to_test[i] for i in [3, 4, 5]])
def test_log_pdf(xscale: Scale):
    normed_test_loc = 0.5
    normed_test_s = 0.1
    test_loc = xscale.denormalize_point(normed_test_loc)
    test_s = normed_test_s * xscale.width

    ergoLogisticMixture = LogisticMixture(
        components=[
            Logistic(
                loc=xscale.denormalize_point(0.2), s=0.5 * xscale.width, scale=xscale,
            ),
            Logistic(loc=test_loc, s=test_s, scale=xscale),
        ],
        probs=[1.8629593e-29, 1.0],
    )
    ergoLogistic = Logistic(loc=test_loc, s=test_s, scale=xscale)

    ## Test PDF
    normed_scipydist = scipy.stats.logistic(normed_test_loc, normed_test_s)
    for x in np.linspace(0, 1, 10):
        denormalized_x = xscale.denormalize_point(x)
        assert (
            normed_scipydist.pdf(x) / xscale.width
            == pytest.approx(float(ergoLogistic.pdf(denormalized_x)), rel=1e-3)
            == pytest.approx(float(ergoLogisticMixture.pdf(denormalized_x)), rel=1e-3)
        )


@pytest.mark.parametrize(
    "fixed_params",
    [{"num_components": 2}, {"num_components": 2, "floor": -0.5, "ceiling": 1.5}],
)
def test_fit_mixture_small(fixed_params):
    xscale = Scale(0, 1)
    mixture = LogisticMixture.from_samples(
        data=np.array([0.1, 0.2, 0.8, 0.9]), fixed_params=fixed_params, scale=xscale,
    )
    for prob in mixture.probs:
        assert prob == pytest.approx(0.5, 0.1)
    locs = sorted([component.base_dist.loc for component in mixture.components])
    assert locs[0] == pytest.approx(0.15, abs=0.1)
    assert locs[1] == pytest.approx(0.85, abs=0.1)


@pytest.mark.parametrize(
    "fixed_params",
    [{"num_components": 2}, {"num_components": 2, "floor": -2, "ceiling": 3}],
)
def test_fit_mixture_large(fixed_params):
    xscale = Scale(-2, 3)
    data1 = onp.random.logistic(loc=0.7, scale=0.1, size=1000)
    data2 = onp.random.logistic(loc=0.4, scale=0.2, size=1000)
    data = onp.concatenate([data1, data2])
    mixture = LogisticMixture.from_samples(
        data=data, fixed_params=fixed_params, scale=xscale,
    )
    # FIXME: What's going on below with scales?
    components = sorted(
        [
            (component.base_dist.loc, component.base_dist.s)
            for component in mixture.components
        ]
    )
    assert components[0][0] == pytest.approx(xscale.normalize_point(0.4), abs=0.2)
    assert components[1][0] == pytest.approx(xscale.normalize_point(0.7), abs=0.2)
    assert components[0][1] == pytest.approx(0.2, abs=0.2)
    assert components[1][1] == pytest.approx(0.1, abs=0.2)


def test_mixture_cdf(logistic_mixture15):
    # Use a mixture with known properties. The median should be 15 for this mixture.
    cdf50 = logistic_mixture15.cdf(15)
    assert cdf50 == pytest.approx(0.5, rel=1e-3)


def test_mixture_ppf(logistic_mixture10):
    # Use a mixtures with known properties. The median should be 10 for this mixture.
    ppf5 = logistic_mixture10.ppf(0.5)
    assert ppf5 == pytest.approx(10, rel=1e-3)


def test_mixture_ppf_adversarial(
    logistic_mixture_p_uneven, logistic_mixture_p_overlapping
):
    # Use a mixture with one very improbable distribution and one dominant distribution
    # Use a mixture with two hugely overlapping distributions

    assert logistic_mixture_p_uneven.ppf(0.5) == pytest.approx(5.0, rel=1e-3)
    assert logistic_mixture_p_uneven.ppf(0.01) == pytest.approx(-17.9755, rel=1e-3)
    assert logistic_mixture_p_uneven.ppf(0.001) == pytest.approx(-29.5337, rel=1e-3)
    assert logistic_mixture_p_uneven.ppf(0.99) == pytest.approx(27.9755, rel=1e-3)
    assert logistic_mixture_p_uneven.ppf(0.999) == pytest.approx(39.5337, rel=1e-3)

    assert logistic_mixture_p_overlapping.ppf(0.5) == pytest.approx(
        4000000.0342351394, rel=1e-3
    )
    assert logistic_mixture_p_overlapping.ppf(0.01) == pytest.approx(
        3080976.018257023, rel=1e-3
    )
    assert logistic_mixture_p_overlapping.ppf(0.001) == pytest.approx(
        2618649.009437881, rel=1e-3
    )
    assert logistic_mixture_p_overlapping.ppf(0.99) == pytest.approx(
        4919024.050213255, rel=1e-3
    )
    assert logistic_mixture_p_overlapping.ppf(0.999) == pytest.approx(
        5381351.059032397, rel=1e-3
    )


def test_ppf_cdf_round_trip():
    mixture = LogisticMixture.from_samples(
        np.array([0.5, 0.4, 0.8, 0.8, 0.9, 0.95, 0.15, 0.1]), {"num_components": 3}
    )
    x = 0.65
    prob = mixture.cdf(x)
    assert mixture.ppf(prob) == pytest.approx(x, rel=1e-3)


@pytest.mark.xfail(reason="Fitting to samples doesn't reliably work yet #219")
def test_fit_samples(logistic_mixture):
    data = np.array([logistic_mixture.sample() for _ in range(0, 1000)])
    fitted_mixture = LogisticMixture.from_samples(data, {"num_components": 2})
    true_locs = sorted([c.loc for c in logistic_mixture.components])
    true_scales = sorted([c.s for c in logistic_mixture.components])
    fitted_locs = sorted([c.loc for c in fitted_mixture.components])
    fitted_scales = sorted([c.s for c in fitted_mixture.components])
    for (true_loc, fitted_loc) in zip(true_locs, fitted_locs):
        assert fitted_loc == pytest.approx(float(true_loc), rel=0.2)
    for (true_scale, fitted_scale) in zip(true_scales, fitted_scales):
        assert fitted_scale == pytest.approx(float(true_scale), rel=0.2)


def test_logistic_mixture_normalization():
    scale = Scale(-50, 50)
    scalex2 = Scale(-100, 100)
    mixture = LogisticMixture(
        components=[Logistic(-40, 1, scale), Logistic(50, 10, scale)], probs=[0.5, 0.5],
    )

    mixturex2 = LogisticMixture(
        components=[Logistic(-80, 2, scalex2), Logistic(100, 20, scalex2)],
        probs=[0.5, 0.5],
    )

    assert mixturex2 == mixture.normalize().denormalize(scalex2)
    assert mixture == mixturex2.normalize().denormalize(scale)

    normalized = (
        mixture.normalize()
    )  # not necessary to normalize but here for readability

    assert normalized == LogisticMixture(
        [Logistic(0.1, 0.01, Scale(0, 1)), Logistic(1, 0.1, Scale(0, 1))], [0.5, 0.5],
    )


def test_destructure(logistic_mixture10, truncated_logistic_mixture):
    for original_mixture in [logistic_mixture10, truncated_logistic_mixture]:
        params = original_mixture.destructure()
        class_params, numeric_params = params
        recovered_mixture = class_params[0].structure(params)
        assert recovered_mixture.pdf(0.5) == pytest.approx(
            float(original_mixture.pdf(0.5))
        )


def test_destructure_with_cond(truncated_logistic_mixture, point_densities):
    PointDensityCondition(
        point_densities["xs"], point_densities["densities"]
    ).describe_fit(truncated_logistic_mixture)
