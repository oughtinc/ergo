"""
Mixtures of Normal distributions

Jax jitting and scipy optimization don't handle classes well so we'll
partially have to work with arrays directly (all the params_*
classmethods).
"""

from dataclasses import dataclass, field
from functools import partial
from typing import List, Type

from jax import grad, jit, nn, scipy, vmap
import jax.numpy as np

from .location_scale_family import Normal
from .mixture import LSMixture


@dataclass
class NormalMixture(LSMixture):
    components: List[Normal]
    probs: List[float]
    component_type: Type[Normal] = field(default=Normal, repr=False)

    ## param-based functions for Jax compatibility ##
    @staticmethod
    def params_gradlogpdf(params, data):
        return _mixture_params_gradlogpdf(params, data)

    @staticmethod
    def params_logpdf(params, data):
        return _mixture_params_logpdf(params, data)

    @staticmethod
    @jit
    def params_logpdf1(params, datum):
        structured_params = params.reshape((-1, 3))
        component_scores = []
        unnormalized_weights = np.array([p[2] for p in structured_params])
        weights = nn.log_softmax(unnormalized_weights)
        for p, weight in zip(structured_params, weights):
            loc = p[0]
            scale = np.max([p[1], 0.01])  # Find a better solution?
            component_scores.append(Normal.params_logpdf(datum, loc, scale) + weight)

        return scipy.special.logsumexp(np.array(component_scores))


@jit
def _mixture_params_logpdf(params, data):
    scores = vmap(partial(NormalMixture.params_logpdf1, params))(data)
    return np.sum(scores)


_mixture_params_gradlogpdf = jit(grad(_mixture_params_logpdf, argnums=0))
