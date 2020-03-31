import numpy as onp
import scipy as oscipy

import jax.numpy as np
from jax import grad, jit, scipy, nn
from jax.experimental.optimizers import clip_grads


def fit_single_scipy(samples):
    with onp.errstate(all='raise'):
        loc, scale = oscipy.stats.logistic.fit(samples)
        scale = min(max(scale, 0.02), 10)
        loc = min(max(loc, -0.1565), 1.1565)
        return loc, scale


def logistic_logpdf(x, loc, scale):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logistic.html
    y = (x - loc) / scale
    return scipy.stats.logistic.logpdf(y) - np.log(scale)


@jit
def mixture_logpdf(data, components):
    score = 0.0
    for datum in data:
        component_scores = []
        unnormalized_weights = np.array(
            [component[2] for component in components])
        weights = nn.log_softmax(unnormalized_weights)
        for component, weight in zip(components, weights):
            loc = component[0]
            scale = abs(component[1])   # Find a better solution?
            component_scores.append(
                logistic_logpdf(
                    datum, loc, scale) + weight)
        score += scipy.special.logsumexp(np.array(component_scores))
    return score


grad_mixture_logpdf = jit(grad(mixture_logpdf, argnums=1))


def initialize_components(num_components):
    # Each component has (location, scale, weight)
    # Weights sum to 1 (are given in log space)
    # Use onp to initialize parameters since we don't want to track randomness
    # here
    components = onp.random.rand(num_components, 3) * 0.1 + 1.
    components[:, 2] = -num_components
    return components


def fit_mixture(data, num_components=3):
    step_size = 0.01
    components = initialize_components(num_components)
    for n in range(1000):
        grads = grad_mixture_logpdf(data, components)
        grads = clip_grads(grads, 1.0)
        if np.any(np.isnan(grads)) or np.any(np.isnan(components)):
            print(grads)
            print(components)
            break
        components = components + step_size * grads
        if n % 100 == 0:
            score = mixture_logpdf(data, components)
            print(f"Log score: {score:.3f}")
    return components


def fit_single(samples):
    components = fit_mixture(samples, num_components=1)
    loc = float(components[0][0])
    scale = float(components[0][1])
    return loc, scale

