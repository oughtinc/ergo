"""
Truncated mixture of logistic distributions
"""
from dataclasses import dataclass
from typing import Sequence

from jax import nn
import jax.numpy as np
import scipy as oscipy

from ergo.scale import Scale

from .logistic import Logistic
from .logistic_mixture import LogisticMixture
from .mixture import Mixture
from .optimizable import Optimizable


@dataclass
class TruncatedLogisticMixture(Mixture, Optimizable):
    components: Sequence[Logistic]
    probs: Sequence[float]
    scale: Scale
    floor: float  # true-scale floor value
    ceiling: float  # true-scale ceiling value

    def __post_init__(self):
        self.normed_floor = self.scale.normalize_point(self.floor)
        self.normed_ceiling = self.scale.normalize_point(self.ceiling)
        self.base_dist = LogisticMixture(self.components, self.probs, self.scale)
        p_below = super().cdf(self.floor)
        p_above = 1 - super().cdf(self.ceiling)
        self.p_inside = 1 - (p_below + p_above)
        self.logp_inside = np.log(self.p_inside)

    def logpdf(self, x):
        # assumes x is normalized
        logp_x = self.base_dist.logpdf(x) - self.logp_inside
        res = np.where(
            x < self.normed_floor,
            -np.inf,
            np.where(x > self.normed_ceiling, -np.inf, logp_x),
        )
        return res

    def cdf(self, x):
        cdf_x = self.base_dist.cdf(x) / self.p_inside
        return np.where(x < self.floor, 0, np.where(x > self.ceiling, 1, cdf_x))

    def ppf(self, q):
        """
        Percent point function (inverse of cdf) at q.
        """
        return oscipy.optimize.bisect(
            lambda x: self.cdf(x) - q,
            self.floor - 1e-9,
            self.ceiling + 1e-9,
            maxiter=1000,
        )

    def sample(self):
        raise NotImplementedError

    @classmethod
    def from_params(cls, fixed_params, opt_params, scale=None, traceable=True):
        # returns a normalized mixture naive to the true distribution
        if scale is None:
            scale = Scale(0, 1)
        floor = fixed_params["floor"]
        ceiling = fixed_params["ceiling"]
        structured_params = opt_params.reshape((-1, 3))
        unnormalized_weights = structured_params[:, 2]
        probs = list(nn.softmax(unnormalized_weights))
        component_dists = [Logistic(p[0], p[1], scale) for p in structured_params]
        return cls(component_dists, probs, scale, floor, ceiling)

    @staticmethod
    def initialize_optimizable_params(fixed_params):
        return LogisticMixture.initialize_optimizable_params(fixed_params)

    @classmethod
    def from_conditions(cls, *args, init_tries=100, opt_tries=2, **kwargs):
        # Increase default initialization and optimization tries
        return super(TruncatedLogisticMixture, cls).from_conditions(
            *args, init_tries=init_tries, opt_tries=opt_tries, **kwargs
        )

    @classmethod
    def from_samples(cls, *args, init_tries=100, opt_tries=2, **kwargs):
        # Increase default initialization and optimization tries
        return super(TruncatedLogisticMixture, cls).from_samples(
            *args, init_tries=init_tries, opt_tries=opt_tries, **kwargs
        )

    def normalize(self):
        normalized_components = [component.normalize() for component in self.components]
        floor = self.scale.normalize_point(self.floor)
        ceiling = self.scale.normalize_point(self.ceiling)
        return self.__class__(
            normalized_components, self.probs, Scale(0, 1), floor, ceiling
        )

    def denormalize(self, scale: Scale):
        denormalized_components = [
            component.denormalize(scale) for component in self.components
        ]
        floor = scale.denormalize_point(self.floor)
        ceiling = scale.denormalize_point(self.ceiling)
        return self.__class__(
            denormalized_components, self.probs, scale, floor, ceiling
        )

    @classmethod
    def normalize_fixed_params(self, fixed_params, scale: Scale):
        norm_fixed_params = dict(fixed_params)
        norm_fixed_params["floor"] = scale.normalize_point(fixed_params["floor"])
        norm_fixed_params["ceiling"] = scale.normalize_point(fixed_params["ceiling"])
        return norm_fixed_params

    def destructure(self):
        scale_cls, scale_params = self.scale.destructure()
        params = (
            [(c.loc, c.s) for c in self.components],
            self.probs,
            (self.floor, self.ceiling),
            scale_params,
        )
        return ((TruncatedLogisticMixture, scale_cls), params)

    @classmethod
    def structure(cls, params):
        component_params, probs, limits, scale_params, scale_cls = params
        scale = scale_cls(*scale_params)
        components = [
            Logistic(l, s, scale, normalized=True) for (l, s) in component_params
        ]
        return cls(
            components=components,
            probs=probs,
            floor=limits[0],
            ceiling=limits[1],
            scale=scale,
        )
