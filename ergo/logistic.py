from dataclasses import dataclass
from pprint import pprint
from typing import List

from jax import grad, jit, nn, scipy, vmap
from jax.experimental.optimizers import clip_grads, sgd
from jax.interpreters.xla import DeviceArray
import jax.numpy as np
import matplotlib.pyplot as pyplot
import numpy as onp
import scipy as oscipy
import seaborn
import torch
from tqdm.autonotebook import tqdm

from ergo.ppl import categorical


@dataclass
class LogisticParams:
    loc: float
    scale: float


@dataclass
class LogisticMixtureParams:
    components: List[LogisticParams]
    probs: List[float]


def fit_single_scipy(samples) -> LogisticParams:
    with onp.errstate(all="raise"):  # type: ignore
        loc, scale = oscipy.stats.logistic.fit(samples)
        return LogisticParams(loc, scale)


@jit
def logistic_logpdf(x, loc, scale) -> DeviceArray:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logistic.html
    y = (x - loc) / scale
    return scipy.stats.logistic.logpdf(y) - np.log(scale)


@jit
def mixture_logpdf_single(datum, components):
    component_scores = []
    unnormalized_weights = np.array([component[2] for component in components])
    weights = nn.log_softmax(unnormalized_weights)
    for component, weight in zip(components, weights):
        loc = component[0]
        scale = np.max([component[1], 0.01])  # Find a better solution?
        component_scores.append(logistic_logpdf(datum, loc, scale) + weight)
    return scipy.special.logsumexp(np.array(component_scores))


@jit
def mixture_logpdf(data, components):
    scores = vmap(lambda datum: mixture_logpdf_single(datum, components))(data)
    return np.sum(scores)


grad_mixture_logpdf = jit(grad(mixture_logpdf, argnums=1))


def initialize_components(num_components):
    # Each component has (location, scale, weight)
    # Weights sum to 1 (are given in log space)
    # We use onp to initialize parameters since we don't want to track
    # randomness
    components = onp.random.rand(num_components, 3) * 0.1 + 1.0
    components[:, 2] = -num_components
    return components


def structure_mixture_params(components) -> LogisticMixtureParams:
    unnormalized_weights = components[:, 2]
    probs = list(np.exp(nn.log_softmax(unnormalized_weights)))
    component_params = [
        LogisticParams(component[0], component[1]) for component in components
    ]
    return LogisticMixtureParams(components=component_params, probs=probs)


def fit_mixture(
    data, num_components=3, verbose=False, num_samples=5000
) -> LogisticMixtureParams:
    # the data might be something weird, like a pandas dataframe column;
    # turn it into a regular old numpy array
    data_as_np_array = np.array(data)
    step_size = 0.01
    components = initialize_components(num_components)
    (init_fun, update_fun, get_params) = sgd(step_size)
    opt_state = init_fun(components)
    for i in tqdm.trange(num_samples):
        components = get_params(opt_state)
        grads = -grad_mixture_logpdf(data_as_np_array, components)
        if np.any(np.isnan(grads)):
            print("Encoutered nan gradient, stopping early")
            print(grads)
            print(components)
            break
        grads = clip_grads(grads, 1.0)
        opt_state = update_fun(i, grads, opt_state)
        if i % 500 == 0 and verbose:
            pprint(components)
            score = mixture_logpdf(data_as_np_array, components)
            print(f"Log score: {score:.3f}")
    return structure_mixture_params(components)


def fit_single(samples) -> LogisticParams:
    params = fit_mixture(samples, num_components=1)
    return params.components[0]


def sample_mixture(mixture_params):
    i = categorical(torch.tensor(mixture_params.probs))
    component_params = mixture_params.components[i]
    return onp.random.logistic(loc=component_params.loc, scale=component_params.scale)


def plot_mixture(params: LogisticMixtureParams, data=None):
    learned_samples = np.array([sample_mixture(params) for _ in range(5000)])
    ax = seaborn.distplot(learned_samples, label="Mixture")
    ax.set(xlabel="Sample value", ylabel="Density")
    if data is not None:
        seaborn.distplot(data, label="Data")
    pyplot.legend()  # type: ignore
    pyplot.show()
