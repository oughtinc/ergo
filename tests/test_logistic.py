import pytest

from ergo.logistic import fit_single_scipy, fit_single

def test_fit_single_scipy():
    loc, scale = fit_single_scipy([0.1, 0.2])
    assert 0.13 < loc < 0.17

def test_fit_single_jax():
    loc, scale = fit_single([0.1, 0.2])
    assert 0.13 < loc < 0.17

def test_fit_single_compare():
    loc1, scale1 = fit_single_scipy([0.1, 0.2])
    loc2, scale2 = fit_single([0.1, 0.2])
    assert abs(loc1 - loc2) < 0.1
    assert abs(scale1 - scale2) < 0.1
