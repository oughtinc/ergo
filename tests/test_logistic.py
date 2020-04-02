import pytest

import numpy as onp
import jax.numpy as np

from ergo.logistic import fit_single_scipy, fit_single, fit_mixture

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

def test_fit_mixture():
    params = fit_mixture(np.array([0.1, 0.2, 0.8, 0.9]), num_components=2)
    for prob in params.probs:
        assert prob == pytest.approx(0.5, 0.05)
    locs = sorted([component.loc for component in params.components])
    assert locs[0] == pytest.approx(0.15, abs=0.05)
    assert locs[1] == pytest.approx(0.85, abs=0.05)
