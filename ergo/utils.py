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


def minimize(*args, init=None, tries=1, verbose=False, **kwargs):
    """
    Wrapper around scipy.optimize.minimize that supports retries
    """
    if "x0" in kwargs:
        raise ValueError("Provide initialization function (init), not x0")
    best_results = None
    best_loss = float("+inf")
    while tries > 0:
        results = oscipy.optimize.minimize(*args, x0=init(), **kwargs)
        tries -= 1
        if best_results is None or results.fun < best_loss or results.success:
            best_results = results
            best_loss = results.fun
        if results.success or tries == 0:
            break
        if verbose:
            print(
                f"Minimize failed (best loss {best_loss}). Retrying ({tries} tries left)..."
            )
    return best_results
