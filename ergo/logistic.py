from dataclasses import dataclass
import itertools
from typing import List

from jax import grad, jit, nn, scipy, vmap
from jax.interpreters.xla import DeviceArray
import jax.numpy as np
import numpy as onp
import scipy as oscipy

from ergo.distributions import categorical


@dataclass
class LogisticParams:
    loc: float
    scale: float

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = max(scale, 0.0000001)  # Do not allow values <= 0

    def __mul__(self, x):
        return LogisticParams(self.loc * x, self.scale * x)


@dataclass
class LogisticMixtureParams:
    components: List[LogisticParams]
    probs: List[float]

    def __mul__(self, x):
        return LogisticMixtureParams(
            [component * x for component in self.components], self.probs
        )


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
def mixture_logpdf(data, component_params):
    components = component_params.reshape((-1, 3))
    scores = vmap(lambda datum: mixture_logpdf_single(datum, components))(data)
    return np.sum(scores)


grad_mixture_logpdf = jit(grad(mixture_logpdf, argnums=1))


def initialize_components(num_components):
    """
    Each component has (location, scale, weight).
    The shape of the components matrix is (num_components, 3).
    Weights sum to 1 (are given in log space).
    We use original numpy to initialize parameters since we don't
    want to track randomness.
    """
    components = onp.random.rand(num_components, 3) * 0.1 + 1.0
    components[:, 2] = -num_components
    component_params = components.reshape(-1)
    bounds = [((None, None), (0.01, None), (None, None)) for _ in range(num_components)]
    bound_params = list(itertools.chain.from_iterable(bounds))
    return component_params, bound_params


def structure_mixture_params(component_params) -> LogisticMixtureParams:
    components = component_params.reshape((-1, 3))
    unnormalized_weights = components[:, 2]
    probs = list(np.exp(nn.log_softmax(unnormalized_weights)))
    component_params = [
        LogisticParams(component[0], component[1]) for component in components
    ]
    return LogisticMixtureParams(components=component_params, probs=probs)


def fit_mixture(data, num_components=3, verbose=False) -> LogisticMixtureParams:
    data = np.array(data)
    z = float(np.mean(data))
    normalized_data = data / z
    component_params, _ = initialize_components(num_components)
    fit_results = oscipy.optimize.minimize(
        lambda comps: -mixture_logpdf(normalized_data, comps),
        x0=component_params,
        jac=lambda comps: -grad_mixture_logpdf(normalized_data, comps),
    )
    if not fit_results.success and verbose:
        print(fit_results)
    component_params = fit_results.x
    return structure_mixture_params(component_params) * z


def fit_single(samples) -> LogisticParams:
    params = fit_mixture(samples, num_components=1)
    return params.components[0]


def sample_mixture(mixture_params):
    i = categorical(np.array(mixture_params.probs))
    component_params = mixture_params.components[i]
    return onp.random.logistic(loc=component_params.loc, scale=component_params.scale)
