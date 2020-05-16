import jax.numpy as np
import numpy as onp
import pytest

from ergo import Logistic, LogisticMixture
import tests.mocks


def test_cdf():
    jax_dist = Logistic(loc=10, scale=1)
    original_dist = jax_dist.rv()
    for x in range(5, 15):
        assert jax_dist.cdf(x) == pytest.approx(original_dist.cdf(x), rel=0.1)


def test_fit_single_scipy():
    dist = Logistic.from_samples_scipy(onp.array([0.1, 0.2]))
    assert dist.loc == pytest.approx(0.15, abs=0.02)


def test_logpdf1():
    assert -18 < tests.mocks.mock_mixture.logpdf1(0.1) < -17


def test_logpdf():
    tests.mocks.mock_mixture.logpdf(np.array([0.1, 0.2]))


def test_fit_single_jax():
    dist = Logistic.from_samples(np.array([0.1, 0.2]))
    assert dist.loc == pytest.approx(0.15, abs=0.02)


def test_fit_single_compare():
    scipy_dist = Logistic.from_samples_scipy(onp.array([0.1, 0.2]))
    jax_dist = Logistic.from_samples(np.array([0.1, 0.2]))
    assert scipy_dist.loc == pytest.approx(float(jax_dist.loc), abs=0.1)
    assert scipy_dist.scale == pytest.approx(float(jax_dist.scale), abs=0.1)


def test_fit_mixture_small():
    mixture = LogisticMixture.from_samples(
        np.array([0.1, 0.2, 0.8, 0.9]), num_components=2
    )
    for prob in mixture.probs:
        assert prob == pytest.approx(0.5, 0.1)
    locs = sorted([component.loc for component in mixture.components])
    assert locs[0] == pytest.approx(0.15, abs=0.1)
    assert locs[1] == pytest.approx(0.85, abs=0.1)


def test_fit_mixture_large():
    data1 = onp.random.logistic(loc=0.7, scale=0.1, size=1000)
    data2 = onp.random.logistic(loc=0.4, scale=0.2, size=1000)
    data = onp.concatenate([data1, data2])
    mixture = LogisticMixture.from_samples(data, num_components=2)
    locs = sorted([component.loc for component in mixture.components])
    scales = sorted([component.scale for component in mixture.components])
    assert locs[0] == pytest.approx(0.4, abs=0.2)
    assert locs[1] == pytest.approx(0.7, abs=0.2)
    assert scales[0] == pytest.approx(0.1, abs=0.2)
    assert scales[1] == pytest.approx(0.2, abs=0.2)


def test_mixture_cdf():
    # make a mock with known properties. The median should be 15 for this mixture
    mock_logistic_params = np.array([[10, 3.658268, 0.5], [20, 3.658268, 0.5]])
    mock_mixture = LogisticMixture.from_params(mock_logistic_params)
    cdf50 = mock_mixture.cdf(15)
    assert cdf50 == pytest.approx(0.5, rel=1e-3)


def test_mixture_ppf():
    # make a mock with known properties. The median should be 10 for this mixture
    mock_logistic_params = np.array([[15, 2.3658268, 0.5], [5, 2.3658268, 0.5]])
    mock_mixture = LogisticMixture.from_params(mock_logistic_params)
    ppf5 = mock_mixture.ppf(0.5)
    assert ppf5 == pytest.approx(10, rel=1e-3)


def ppf_cdf_round_trip():
    mock_mixture = LogisticMixture.from_samples(
        np.array([0.5, 0.4, 0.8, 0.8, 0.9, 0.95, 0.15, 0.1]), num_components=3
    )
    x = 0.65
    prob = mock_mixture.cdf(x)
    assert mock_mixture.ppf(prob) == pytest.approx(x, rel=1e-3)


def test_fit_samples():
    mock_mixture = tests.mocks.mock_mixture
    data = np.array([mock_mixture.sample() for _ in range(0, 1000)])
    fitted_mixture = LogisticMixture.from_samples(data, num_components=2)
    true_locs = sorted([c.loc for c in mock_mixture.components])
    true_scales = sorted([c.scale for c in mock_mixture.components])
    fitted_locs = sorted([c.loc for c in fitted_mixture.components])
    fitted_scales = sorted([c.scale for c in fitted_mixture.components])
    for (true_loc, fitted_loc) in zip(true_locs, fitted_locs):
        assert fitted_loc == pytest.approx(float(true_loc), rel=0.2)
    for (true_scale, fitted_scale) in zip(true_scales, fitted_scales):
        assert fitted_scale == pytest.approx(float(true_scale), rel=0.2)
