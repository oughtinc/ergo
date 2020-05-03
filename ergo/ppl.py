"""
This module provides a few lightweight wrappers around probabilistic
programming primitives from Numpyro.
"""

import functools
from typing import Dict, List

import jax
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
    if not name:
        # Values that aren't explicitly named
        name = "_v"
    # The rng key provided below is only used when no Numpyro seed handler
    # is provided. This happens when we sample from distributions outside
    # an inference context.
    return numpyro.sample(name, dist, rng_key=onetime_rng_key(), **kwargs)


# Marking deterministic values


def tag(value, name: str):
    return numpyro.deterministic(name, value)


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


# Marking function output as a particular deterministic value


def tag_output(model):
    def wrapped():
        value = model()
        if value is not None:
            tag(value, "output")
        return value

    return wrapped


# Main inference function


def run(model, num_samples=5000, ignore_untagged=True, rng_seed=0) -> pd.DataFrame:
    """
    Run model forward, record samples for variables. Return dataframe
    with one row for each execution.
    """
    model = numpyro.handlers.trace(handle_mem(tag_output(autoname(model))))
    with numpyro.handlers.seed(rng_seed=rng_seed):
        samples: List[Dict[str, float]] = []
        for _ in tqdm(range(num_samples)):
            sample: Dict[str, float] = {}
            trace = model.get_trace()
            for name in trace.keys():
                if trace[name]["type"] in ("sample", "deterministic"):
                    if ignore_untagged and name.startswith("_"):
                        continue
                    sample[name] = trace[name]["value"]
            samples.append(sample)
    return pd.DataFrame(samples)  # type: ignore
