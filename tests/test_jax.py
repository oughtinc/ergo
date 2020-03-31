import pytest
import jax
import jax.numpy as np
from jax import grad, jit


def f(x, y):
    return x * 2.0 + y * 3.0


def test_jax():
    gf = jit(grad(f, (0, 1)))
    grads = gf(1., 1.)
    assert float(grads[0]) == pytest.approx(2.0)
    assert float(grads[1]) == pytest.approx(3.0)


test_jax()
