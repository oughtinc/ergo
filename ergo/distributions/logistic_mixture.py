"""
Mixture of logistic distributions
"""
from dataclasses import dataclass
import itertools
from typing import Sequence

from jax import nn
import jax.numpy as np
import numpy as onp

from ergo.scale import Scale
import ergo.static as static

from .logistic import Logistic
from .mixture import Mixture
from .optimizable import Optimizable


@dataclass
class LogisticMixture(Mixture, Optimizable):
    components: Sequence[Logistic]
    probs: Sequence[float]
    scale: Scale

    def logpdf(self, x):
        # assumes x is normalized
        nested_params = [
            [c.loc, c.s, weight] for c, weight in zip(self.components, self.probs)
        ]
        opt_params = np.array(list(itertools.chain.from_iterable(nested_params)))

        return static.logistic_mixture_logpdf(opt_params, x)

    @classmethod
    def from_params(cls, fixed_params, opt_params, scale=None, traceable=True):
        # returns a normalized mixture naive to the true distribution
        if scale is None:
            scale = Scale(0, 1)
        structured_params = opt_params.reshape((-1, 3))
        unnormalized_weights = structured_params[:, 2]
        probs = list(nn.softmax(unnormalized_weights))
        component_dists = [Logistic(p[0], p[1], scale) for p in structured_params]
        return cls(component_dists, probs, scale)

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
        weights = onp.full(num_components, -float(num_components))
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

    @classmethod
    def normalize_fixed_params(self, fixed_params, scale: Scale):
        # no normalization required
        return fixed_params

    def destructure(self):
        scale_cls, scale_params = self.scale.destructure()
        params = (
            [(c.loc, c.s) for c in self.components],
            self.probs,
            scale_params,
        )
        return ((LogisticMixture, scale_cls), params)

    @classmethod
    def structure(cls, params):
        component_params, probs, scale_params, scale_cls = params
        scale = scale_cls(*scale_params)
        components = [
            Logistic(l, s, scale, normalized=True) for (l, s) in component_params
        ]
        return cls(components=components, probs=probs, scale=scale)
