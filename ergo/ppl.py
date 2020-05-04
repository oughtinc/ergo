"""
This module provides a few lightweight wrappers around probabilistic
programming primitives from Numpyro.
"""

import functools
from typing import Dict, List

import jax
import jax.numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from tqdm.autonotebook import tqdm

from ergo.autoname import autoname

# Random numbers

_RNG_KEY = jax.random.PRNGKey(0)


def onetime_rng_key():
    global _RNG_KEY
    current_key, _RNG_KEY = jax.random.split(_RNG_KEY, 2)
    return current_key


# Sampling from probability distributions


def sample(dist: dist.Distribution, name: str = None, **kwargs):
    """
    Sample from a primitive distribution

    :param dist: A Pyro distribution
    :param name: Name to assign to this sampling site in the execution trace
    :return: A sample from the distribution
    """
    # If a value isn't explicitly named, generate an automatic name,
    # relying on autoname handler for uniqueness.
    if not name:
        name = "_v"
    # The rng key provided below is only used when no Numpyro seed handler
    # is provided. This happens when we sample from distributions outside
    # an inference context.
    return numpyro.sample(name, dist, rng_key=onetime_rng_key(), **kwargs)


# Conditioning


def condition(cond: bool, name: str = None):
    if not name:
        name = "_c"
    return numpyro.factor(name, 0 if cond else np.NINF)


# Record deterministic values in trace


def tag(value, name: str):
    return numpyro.deterministic(name, value)


# Automatically record model return value in trace


def tag_output(model):
    def wrapped():
        value = model()
        if value is not None:
            tag(value, "output")
        return value

    return wrapped


# Memoization

memoized_functions = []  # FIXME: global state


def mem(func):
    func = functools.lru_cache(None)(func)
    memoized_functions.append(func)
    return func


def clear_mem():
    for func in memoized_functions:
        func.cache_clear()


def handle_mem(model):
    def wrapped():
        clear_mem()
        return model()

    return model


# Main inference function


def is_singleton_array(value):
    return isinstance(value, np.DeviceArray) and value.size in ((1,), 1)


def is_factor(entry):
    return entry["is_observed"] and isinstance(entry["fn"], numpyro.distributions.Unit)


def factor_score(entry):
    return entry["fn"].log_factor


def run(model, num_samples=5000, ignore_untagged=True, rng_seed=0) -> pd.DataFrame:
    """
    Run model forward, record samples for variables. Return dataframe
    with one row for each execution.
    """
    model = numpyro.handlers.trace(handle_mem(tag_output(autoname(model))))
    with numpyro.handlers.seed(rng_seed=rng_seed):
        samples: List[Dict[str, float]] = []
        progress_bar = tqdm(total=num_samples)
        i = 0
        while i < num_samples:
            sample: Dict[str, float] = {}
            trace = model.get_trace()
            reject = False
            for name in trace.keys():
                entry = trace[name]
                if entry["type"] in ("sample", "deterministic"):
                    if is_factor(entry):
                        score = factor_score(entry)
                        if score == np.NINF:
                            reject = True
                            break
                        elif score == 0:
                            pass
                        else:
                            raise NotImplementedError(
                                f"Weighted factors - got score {score}"
                            )
                    else:
                        if ignore_untagged and name.startswith("_"):
                            continue
                        value = entry["value"]
                        if is_singleton_array(value):
                            value = value.item()  # FIXME
                        sample[name] = value
            if reject:
                continue
            samples.append(sample)
            i += 1
            progress_bar.update(i)
        progress_bar.close()

    return pd.DataFrame(samples)  # type: ignore
