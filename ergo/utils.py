import functools
import weakref

import jax.numpy as np
import scipy as oscipy


def to_float(value):
    """Convert value to float"""
    return np.asscalar(value)


def memoized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator


def minimize_random(fun, init, tries=100):
    best_x = None
    best_loss = float("+inf")
    while tries > 0:
        x = init()
        loss = fun(x)
        if best_x is None or loss < best_loss:
            best_x = x
            best_loss = loss
        tries -= 1
    return best_x


def minimize(fun, *args, init=None, init_tries=1, opt_tries=1, verbose=False, **kwargs):
    """
    Wrapper around scipy.optimize.minimize that supports retries
    """
    if "x0" in kwargs:
        raise ValueError("Provide initialization function (init), not x0")

    best_results = None
    best_loss = float("+inf")
    while opt_tries > 0:
        init_params = minimize_random(fun, init, tries=init_tries)
        results = oscipy.optimize.minimize(fun, *args, x0=init_params, **kwargs)
        opt_tries -= 1
        if best_results is None or results.fun < best_loss:
            best_results = results
            best_loss = results.fun
        if opt_tries == 0:
            break
    return best_results


def shift(xs, k, fill_value):
    return np.concatenate((np.full(k, fill_value), xs[:-k]))

# Taken form https://github.com/google/jax/pull/3042/files (has been merged but not released)
def trapz(y, x=None, dx=1.0):
    if x is not None:
        dx = np.diff(x)
    return 0.5 * (dx * (y[..., 1:] + y[..., :-1])).sum(-1)
