"""
This module provides a few lightweight wrappers around probabilistic
programming primitives from Numpyro.
"""

import functools
import math
from typing import Dict, List

import jax.numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from tqdm.autonotebook import tqdm

from ergo.autoname import autoname


def to_float(value):
    """Convert value to float"""
    return np.asscalar(value)


# Core functionality


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
    return numpyro.sample(name, dist, **kwargs)


def tag(value, name: str):
    return numpyro.deterministic(name, value)


# Provide samplers for primitive distributions


def bernoulli(p, **kwargs):
    return sample(dist.Bernoulli(probs=p), **kwargs)


def normal(mean=0, stdev=1, **kwargs):
    return sample(dist.Normal(mean, stdev), **kwargs)


def lognormal(mean=0, stdev=1, **kwargs):
    return sample(dist.LogNormal(mean, stdev), **kwargs)


def halfnormal(stdev, **kwargs):
    return sample(dist.HalfNormal(stdev), **kwargs)


def uniform(low=0, high=1, **kwargs):
    return sample(dist.Uniform(low, high), **kwargs)


def beta(a=1, b=1, **kwargs):
    return sample(dist.Beta(a, b), **kwargs)


def categorical(ps, **kwargs):
    return sample(dist.Categorical(ps), **kwargs)


# Provide alternative parameterizations for primitive distributions


def NormalFromInterval(low, high):
    """This assumes a centered 90% confidence interval, i.e. the left endpoint
    marks 0.05% on the CDF, the right 0.95%."""
    mean = (high + low) / 2
    stdev = (high - mean) / 1.645
    return dist.Normal(mean, stdev)


def HalfNormalFromInterval(high):
    """This assumes a 90% confidence interval starting at 0,
    i.e. right endpoint marks 90% on the CDF"""
    stdev = high / 1.645
    return dist.HalfNormal(stdev)


def LogNormalFromInterval(low, high):
    """This assumes a centered 90% confidence interval, i.e. the left endpoint
    marks 0.05% on the CDF, the right 0.95%."""
    loghigh = math.log(high)
    loglow = math.log(low)
    mean = (loghigh + loglow) / 2
    stdev = (loghigh - loglow) / (2 * 1.645)
    return dist.LogNormal(mean, stdev)


def BetaFromHits(hits, total):
    return dist.Beta(hits, (total - hits))


# Alternative names and parameterizations for primitive distribution samplers


def normal_from_interval(low, high, **kwargs):
    return sample(NormalFromInterval(low, high), **kwargs)


def lognormal_from_interval(low, high, **kwargs):
    return sample(LogNormalFromInterval(low, high), **kwargs)


def halfnormal_from_interval(high, **kwargs):
    return sample(HalfNormalFromInterval(high), **kwargs)


def beta_from_hits(hits, total, **kwargs):
    return sample(BetaFromHits(hits, total), **kwargs)


def random_choice(options, ps=None):
    if ps is None:
        ps = np.full(len(options), 1 / len(options))
    else:
        ps = np.array(ps)

    idx = sample(dist.Categorical(ps))
    return options[idx]


def random_integer(min: int, max: int, **kwargs) -> int:
    return int(math.floor(uniform(min, max, **kwargs).item()))


flip = bernoulli


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


# Inference


def tag_output(model):
    def wrapped():
        value = model()
        if value is not None:
            tag(value, "output")
        return value

    return wrapped


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
