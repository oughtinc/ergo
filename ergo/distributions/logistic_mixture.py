"""
Mixtures of logistic distributions

Jax jitting and scipy optimization don't handle classes well so we'll
partially have to work with arrays directly (all the params_*
classmethods).
"""

from dataclasses import dataclass
from functools import partial
import itertools
from typing import List

from jax import grad, jit, nn, scipy, vmap
import jax.numpy as np
import numpy as onp

from .location_scale_family import Logistic
from .mixture import Mixture


@dataclass
class LogisticMixture(Mixture):
    components: List[Logistic]
    probs: List[float]

    def __mul__(self, x):
        return LogisticMixture(
            [component * x for component in self.components], self.probs
        )

    @staticmethod
    def initialize_params(num_components):
        """
        Each component has (location, scale, weight).
        The shape of the components matrix is (num_components, 3).
        Weights sum to 1 (are given in log space).
        We use original numpy to initialize parameters since we don't
        want to track randomness.
        """
        components = onp.random.rand(num_components, 3) * 0.1 + 1.0
        components[:, 2] = -num_components
        return components.reshape(-1)

    @classmethod
    def from_params(cls, params):
        structured_params = params.reshape((-1, 3))
        unnormalized_weights = structured_params[:, 2]
        probs = list(np.exp(nn.log_softmax(unnormalized_weights)))
        component_dists = [Logistic(p[0], p[1]) for p in structured_params]
        return cls(component_dists, probs)

    def to_params(self):
        nested_params = [
            [c.loc, c.scale, weight] for c, weight in zip(self.components, self.probs)
        ]
        return np.array(list(itertools.chain.from_iterable(nested_params)))

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
            component_scores.append(Logistic.params_logpdf(datum, loc, scale) + weight)

        return scipy.special.logsumexp(np.array(component_scores))


@jit
def _mixture_params_logpdf(params, data):
    scores = vmap(partial(LogisticMixture.params_logpdf1, params))(data)
    return np.sum(scores)


_mixture_params_gradlogpdf = jit(grad(_mixture_params_logpdf, argnums=0))
