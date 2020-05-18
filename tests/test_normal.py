import jax.numpy as np
import numpy as onp
import pytest

from ergo import Normal, NormalMixture


def test_cdf():
    jax_dist = Normal(loc=10, scale=1)
    original_dist = jax_dist.rv()
    for x in range(5, 15):
        assert jax_dist.cdf(x) == pytest.approx(original_dist.cdf(x), rel=0.1)


def test_fit_single_scipy():
    dist = Normal.from_samples_scipy(onp.array([0.1, 0.2]))
    assert dist.loc == pytest.approx(0.15, abs=0.02)


def test_logpdf1(normal_mixture):
    assert -59 < normal_mixture.logpdf1(0.1) < -58


def test_logpdf(normal_mixture):
    normal_mixture.logpdf(np.array([0.1, 0.2]))


def test_fit_single_jax():
    dist = Normal.from_samples(np.array([0.1, 0.2]))
    assert dist.loc == pytest.approx(0.15, abs=0.02)


def test_fit_single_compare():
    scipy_dist = Normal.from_samples_scipy(onp.array([0.1, 0.2]))
    jax_dist = Normal.from_samples(np.array([0.1, 0.2]))
    assert scipy_dist.loc == pytest.approx(float(jax_dist.loc), abs=0.1)
    assert scipy_dist.scale == pytest.approx(float(jax_dist.scale), abs=0.1)


def test_fit_mixture_small():
    mixture = NormalMixture.from_samples(
        np.array([0.1, 0.2, 0.8, 0.9]), num_components=2
    )
    for prob in mixture.probs:
        assert prob == pytest.approx(0.5, 0.1)
    locs = sorted([component.loc for component in mixture.components])
    assert locs[0] == pytest.approx(0.15, abs=0.1)
    assert locs[1] == pytest.approx(0.85, abs=0.1)


def test_fit_mixture_large():
    data1 = onp.random.norm(loc=0.7, scale=0.1, size=1000)
    data2 = onp.random.norm(loc=0.4, scale=0.2, size=1000)
    data = onp.concatenate([data1, data2])
    mixture = NormalMixture.from_samples(data, num_components=2)
    locs = sorted([component.loc for component in mixture.components])
    scales = sorted([component.scale for component in mixture.components])
    assert locs[0] == pytest.approx(0.4, abs=0.2)
    assert locs[1] == pytest.approx(0.7, abs=0.2)
    assert scales[0] == pytest.approx(0.1, abs=0.2)
    assert scales[1] == pytest.approx(0.2, abs=0.2)


def test_mixture_cdf():
    # Make a mixture with known properties. The median should be 15 for this mixture.
    normal_params = np.array([[10, 3.658268, 0.5], [20, 3.658268, 0.5]])
    mixture = NormalMixture.from_params(normal_params)
    cdf50 = mixture.cdf(15)
    assert cdf50 == pytest.approx(0.5, rel=1e-3)


def test_mixture_ppf():
    # Make a mixtures with known properties. The median should be 10 for this mixture.
    normal_params = np.array([[15, 2.3658268, 0.5], [5, 2.3658268, 0.5]])
    mixture = NormalMixture.from_params(normal_params)
    ppf5 = mixture.ppf(0.5)
    assert ppf5 == pytest.approx(10, rel=1e-3)


def ppf_cdf_round_trip():
    mixture = NormalMixture.from_samples(
        np.array([0.5, 0.4, 0.8, 0.8, 0.9, 0.95, 0.15, 0.1]), num_components=3
    )
    x = 0.65
    prob = mixture.cdf(x)
    assert mixture.ppf(prob) == pytest.approx(x, rel=1e-3)


def test_fit_samples(normal_mixture):
    data = np.array([normal_mixture.sample() for _ in range(0, 1000)])
    fitted_mixture = NormalMixture.from_samples(data, num_components=2)
    true_locs = sorted([c.loc for c in normal_mixture.components])
    true_scales = sorted([c.scale for c in normal_mixture.components])
    fitted_locs = sorted([c.loc for c in fitted_mixture.components])
    fitted_scales = sorted([c.scale for c in fitted_mixture.components])
    for (true_loc, fitted_loc) in zip(true_locs, fitted_locs):
        assert fitted_loc == pytest.approx(float(true_loc), rel=0.2)
    for (true_scale, fitted_scale) in zip(true_scales, fitted_scales):
        assert fitted_scale == pytest.approx(float(true_scale), rel=0.2)
