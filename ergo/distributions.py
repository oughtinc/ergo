"""
This module provides samplers for probability distributions.
"""

import math

import jax.numpy as np
import numpyro.distributions as dist

from ergo.ppl import sample


def bernoulli(p=0.5, **kwargs):
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
