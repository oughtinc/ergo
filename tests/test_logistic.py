import jax.numpy as np
import numpy as onp
import pytest

from ergo.logistic import (
    fit_mixture,
    fit_single,
    fit_single_scipy,
    mixture_cdf,
    mixture_ppf,
    sample_mixture,
    structure_mixture_params,
)
import tests.mocks


def test_fit_single_scipy():
    params = fit_single_scipy(onp.array([0.1, 0.2]))
    assert params.loc == pytest.approx(0.15, abs=0.02)


def test_fit_single_jax():
    params = fit_single(np.array([0.1, 0.2]))
    assert params.loc == pytest.approx(0.15, abs=0.02)


def test_fit_single_compare():
    scipy_params = fit_single_scipy(onp.array([0.1, 0.2]))
    jax_params = fit_single(np.array([0.1, 0.2]))
    assert scipy_params.loc == pytest.approx(jax_params.loc, abs=0.1)
    assert scipy_params.scale == pytest.approx(jax_params.scale, abs=0.1)


def test_fit_mixture_small():
    params = fit_mixture(np.array([0.1, 0.2, 0.8, 0.9]), num_components=2)
    for prob in params.probs:
        assert prob == pytest.approx(0.5, 0.1)
    locs = sorted([component.loc for component in params.components])
    assert locs[0] == pytest.approx(0.15, abs=0.1)
    assert locs[1] == pytest.approx(0.85, abs=0.1)


def test_fit_mixture_large():
    data1 = onp.random.logistic(loc=0.7, scale=0.1, size=1000)
    data2 = onp.random.logistic(loc=0.4, scale=0.2, size=1000)
    data = onp.concatenate([data1, data2])
    params = fit_mixture(data, num_components=2)
    locs = sorted([component.loc for component in params.components])
    scales = sorted([component.scale for component in params.components])
    assert locs[0] == pytest.approx(0.4, abs=0.2)
    assert locs[1] == pytest.approx(0.7, abs=0.2)
    assert scales[0] == pytest.approx(0.1, abs=0.2)
    assert scales[1] == pytest.approx(0.2, abs=0.2)


def test_mixture_cdf():
    # make a mock with known properties. The median should be 15 for this mixture
    mock_logistic_params = np.array([[10, 3.658268, 0.5], [20, 3.658268, 0.5]])
    mock_mixture = structure_mixture_params(mock_logistic_params)
    cdf50 = mixture_cdf(15, mock_mixture)
    assert cdf50 == pytest.approx(0.5, rel=1e-3)


def test_mixture_ppf():
    # make a mock with known properties. The median should be 10 for this mixture
    mock_logistic_params = np.array([[15, 2.3658268, 0.5], [5, 2.3658268, 0.5]])
    mock_mixture = structure_mixture_params(mock_logistic_params)
    ppf5 = mixture_ppf(0.5, mock_mixture)
    assert ppf5 == pytest.approx(10, rel=1e-3)


def ppf_cdf_round_trip():
    mock_mixture = fit_mixture(
        np.array([0.5, 0.4, 0.8, 0.8, 0.9, 0.95, 0.15, 0.1]), num_components=3
    )
    x = 0.65
    prob = mixture_cdf(x, mock_mixture)
    assert mixture_ppf(prob, mock_mixture) == pytest.approx(x, rel=1e-3)


def test_fit_samples():
    data = np.array(
        [sample_mixture(tests.mocks.mock_true_params) for _ in range(0, 1000)]
    )
    true_params = tests.mocks.mock_true_params
    fitted_params = fit_mixture(data, num_components=2)
    true_locs = sorted([component.loc for component in true_params.components])
    true_scales = sorted([component.scale for component in true_params.components])
    fitted_locs = sorted([component.loc for component in fitted_params.components])
    fitted_scales = sorted([component.scale for component in fitted_params.components])
    for (true_loc, fitted_loc) in zip(true_locs, fitted_locs):
        assert fitted_loc == pytest.approx(true_loc, rel=0.2)
    for (true_scale, fitted_scale) in zip(true_scales, fitted_scales):
        assert fitted_scale == pytest.approx(true_scale, rel=0.2)
