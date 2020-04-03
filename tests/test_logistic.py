import pytest

import numpy as onp
import jax.numpy as np
import numpy as onp

from ergo.logistic import fit_single_scipy, fit_single, fit_mixture, plot_mixture


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

# visual tests, comment out usually

# def test_visual_plot_mixture():
#     samples = onp.concatenate(
#         [onp.random.normal(loc=0.1, scale=0.1, size=1500),
#          onp.random.normal(loc=0.5, scale=0.1, size=1500),
#          onp.random.logistic(loc=0.9, scale=0.1, size=2000)])
#     onp.random.shuffle(samples)
#     mixture_params = fit_mixture(samples, num_components=5, num_samples=1000)
#     plot_mixture(mixture_params, samples)
