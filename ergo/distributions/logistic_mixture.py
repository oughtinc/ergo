"""
Mixture of logistic distributions
"""
from dataclasses import dataclass
import itertools
from typing import Sequence

from jax import nn
import jax.numpy as np
import numpy as onp

import ergo.static as static

from .logistic import Logistic
from .mixture import Mixture
from .optimizable import Optimizable


@dataclass
class LogisticMixture(Mixture, Optimizable):
    components: Sequence[Logistic]
    probs: Sequence[float]

    def logpdf(self, x):
        nested_params = [
            [c.loc, c.scale, weight] for c, weight in zip(self.components, self.probs)
        ]
        opt_params = np.array(list(itertools.chain.from_iterable(nested_params)))
        return static.logistic_mixture_logpdf(opt_params, x)

    @classmethod
    def from_params(cls, fixed_params, opt_params, traceable=True):
        structured_params = opt_params.reshape((-1, 3))
        locs = structured_params[:, 0]
        scales = np.abs(structured_params[:, 1])
        probs = list(nn.softmax(structured_params[:, 2]))
        component_dists = [Logistic(l, s) for (l, s) in zip(locs, scales)]
        return cls(component_dists, probs)

    @staticmethod
    def initialize_optimizable_params(fixed_params):
        """
        Each component has (location, scale, weight).
        The shape of the components matrix is (num_components, 3).
        Weights sum to 1 (are given in log space).
        We use original numpy to initialize parameters since we don't
        want to track randomness.
        """
        num_components = fixed_params["num_components"]
        scale_multiplier = 0.2
        locs = onp.random.rand(num_components)
        scales = onp.random.rand(num_components) * scale_multiplier
        weights = onp.full(num_components, -num_components)
        components = onp.stack([locs, scales, weights]).transpose()
        return components.reshape(-1)

    @classmethod
    def from_conditions(cls, *args, init_tries=100, opt_tries=2, **kwargs):
        # Increase default initialization and optimization tries
        return super(LogisticMixture, cls).from_conditions(
            *args, init_tries=init_tries, opt_tries=opt_tries, **kwargs
        )

    @classmethod
    def from_samples(cls, *args, init_tries=100, opt_tries=2, **kwargs):
        # Increase default initialization and optimization tries
        return super(LogisticMixture, cls).from_samples(
            *args, init_tries=init_tries, opt_tries=opt_tries, **kwargs
        )

    def destructure(self):
        params = ([(c.loc, c.scale) for c in self.components], self.probs)
        return (LogisticMixture, params)

    @classmethod
    def structure(cls, params):
        component_params, probs = params
        components = [Logistic(l, s) for (l, s) in component_params]
        return cls(components=components, probs=probs)
