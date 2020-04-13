import math

import pyro
import torch
import tqdm

import pandas as pd
import pyro.distributions as dist  # type: ignore

from pyro.contrib.autoname import name_count
from pyro.infer import SVI, Trace_ELBO, Predictive  # type: ignore

from typing import Dict, List


# Config

pyro.enable_validation(True)


# Core functionality


def sample(dist: dist.Distribution, name: str = None, **kwargs):
    if not name:
        # If no name is provided, the model should use the @name_count
        # decorator to avoid using the same name for multiple variables
        name = "_var"
    return pyro.sample(name, dist, **kwargs)


def tag(value, name: str):
    if not isinstance(value, torch.Tensor):
        value = torch.Tensor(value)  # type: ignore
    return pyro.deterministic(name, value)


def to_float(value):
    """Convert value to float"""
    return torch.tensor(value).type(torch.float)


# Provide samplers for primitive distributions


def bernoulli(p, **kwargs):
    return sample(dist.Bernoulli(probs=p), **kwargs)


def normal(mean=0, stdev=1, **kwargs):
    return sample(dist.Normal(mean, stdev), **kwargs)


def lognormal(mean=0, stdev=1, **kwargs):
    return sample(dist.LogNormal(mean, stdev), **kwargs)


def uniform(low=0, high=1, **kwargs):
    return sample(dist.Uniform(low, high), **kwargs)


def beta(alpha=1, beta=1, **kwargs):
    return sample(dist.Beta(alpha, beta), **kwargs)


def categorical(ps, **kwargs):
    return sample(dist.Categorical(ps), **kwargs)


# Provide alternative parameterizations for primitive distributions


def NormalFromInterval(low, high):
    # This assumes a centered 90% confidence interval, i.e. the left endpoint
    # marks 0.05% on the CDF, the right 0.95%.
    mean = (high + low) / 2
    stdev = (high - mean) / 1.645
    return dist.Normal(mean, stdev)


def HalfNormalFromInterval(high):
    # This assumes a 90% confidence interval starting at 0,
    # i.e. right endpoint marks 90% on the CDF
    stdev = high / 1.645
    return dist.HalfNormal(stdev)


def LogNormalFromInterval(low, high):
    # This assumes a centered 90% confidence interval, i.e. the left endpoint
    # marks 0.05% on the CDF, the right 0.95%.
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


def random_choice(options, **kwargs):
    ps = torch.Tensor([1 / len(options)] * len(options))
    idx = sample(dist.Categorical(ps))
    return options[idx]


def random_integer(min: int, max: int, **kwargs) -> int:
    return int(math.floor(uniform(min, max, **kwargs).item()))


flip = bernoulli


# Stats


def run(model, num_samples=5000, ignore_unnamed=True) -> pd.DataFrame:
    """
    1. Run model forward, record samples for variables
    2. Return dataframe with one row for each execution
    """
    model = name_count(model)
    samples: Dict[str, List[float]] = {}
    for i in tqdm.trange(num_samples):
        trace = pyro.poutine.trace(model).get_trace()
        for name in trace.nodes.keys():
            if trace.nodes[name]["type"] == "sample":
                if not ignore_unnamed or not name.startswith("_var"):
                    samples.setdefault(name, [])
                    samples[name].append(trace.nodes[name]["value"].item())  # FIXME
    return pd.DataFrame(samples)  # type: ignore


def infer_and_run(
    model,
    num_samples=5000,
    num_iterations=2000,
    debug=False,
    learning_rate=0.01,
    early_stopping_patience=200,
) -> pd.DataFrame:
    """ 
    debug - whether to output debug information
    num_iterations - Number of optimizer iterations
    learning_rate - Optimizer learning rate
    early_stopping_patience - Stop training if loss hasn't improved for this many iterations
  """

    def to_numpy(d):
        return {k: v.detach().numpy() for k, v in d.items()}

    def debug_output(guide):
        quantiles = to_numpy(guide.quantiles([0.05, 0.5, 0.95]))
        for k, v in quantiles.items():
            print(f"{k}: {v[1]:.4f} [{v[0]:.4f}, {v[2]:.4f}]")

    model = name_count(model)
    
    # Automatically chooses a normal distribution for each variable
    guide = pyro.infer.autoguide.AutoNormal(
        model, init_loc_fn=pyro.infer.autoguide.init_to_median
    )
    pyro.clear_param_store()

    if debug:
        guide(training=True)  # Needed to initialize the guide before output
        debug_output(guide)
        print()

    adam = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    best_loss = None
    last_improvement = None

    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(training=True)
        if best_loss is None or best_loss > loss:
            best_loss = loss
            last_improvement = j
        if j % 100 == 0:
            if debug:
                print("[iteration %04d]" % (j + 1))
                print(f"loss: {loss:.4f}")
                debug_output(guide)
                print()
            if j > (last_improvement + early_stopping_patience):
                print("Stopping Early")
                break

    print(f"Final loss: {loss:.4f}")
    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    raw_samples = predictive(training=False)
    return pd.DataFrame(to_numpy(raw_samples))

