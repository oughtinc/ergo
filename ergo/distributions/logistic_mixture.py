from dataclasses import dataclass
from typing import Sequence

from jax import nn, scipy
import jax.numpy as np
import numpy as onp
import scipy as oscipy

from ergo.scale import Scale

from .base import categorical
from .distribution import Distribution
from .logistic import Logistic
from .optimizable import Optimizable
from .truncate import Truncate


@dataclass
class LogisticMixture(Distribution, Optimizable):
    """
    Mixture of logistic distributions as used by Metaculus

    Metaculus mixture weights apply to the truncated renormalized
    logistics, not to the non-truncated prior.
    """

    components: Sequence[Logistic]
    probs: Sequence[float]

    # Distribution

    def pdf(self, x):
        x = np.asarray(x)
        # last dimension will be component-wise pdf values
        unscaled_pdfs = np.stack([c.pdf(x) for c in self.components], axis=-1)
        # multiply probs and sum by last dimension
        return np.dot(unscaled_pdfs, np.asarray(self.probs))

    def logpdf(self, x):
        x = np.asarray(x)
        log_pdfs = np.stack([c.logpdf(x) for c in self.components], axis=-1)
        log_probs = np.log(np.asarray(self.probs))
        scores = np.add(log_pdfs, log_probs)
        return scipy.special.logsumexp(scores, axis=-1)

    def cdf(self, x):
        x = np.asarray(x)
        unscaled_cdfs = np.stack([c.cdf(x) for c in self.components], axis=-1)
        return np.dot(unscaled_cdfs, np.asarray(self.probs))

    def ppf(self, qs):
        """
        Percent point function (inverse of cdf) at q.

        Returns the smallest x where the mixture_cdf(x) is greater
        than the requested q provided:

            argmin{x} where mixture_cdf(x) > q

        The quantile of a mixture distribution can always be found
        within the range of its components quantiles:
        https://cran.r-project.org/web/packages/mistr/vignettes/mistr-introduction.pdf
        """
        if len(self.components) == 1:
            return self.components[0].ppf(qs)
        ppfs = np.stack([c.ppf(qs) for c in self.components], axis=-1)
        cmins = np.amin(ppfs, axis=-1)
        cmaxs = np.amax(ppfs, axis=-1)

        def find_cdf(q, cmin, cmax):
            return oscipy.optimize.bisect(
                lambda x: self.cdf(x) - q,
                cmin - abs(cmin / 100),
                cmax + abs(cmax / 100),
                maxiter=1000,
            )

        # note that this is not substancially faster than calling it one element at a time
        return np.vectorize(find_cdf)(np.asarray(qs), cmins, cmaxs)

    def sample(self):
        i = categorical(np.array(self.probs))
        component_dist = self.components[i]
        return component_dist.sample()

    # Scaled

    @property
    def scale(self):
        # We require that all scales are the same
        return self.components[0].scale

    def normalize(self):
        normed_components = [c.normalize() for c in self.components]
        return self.__class__(components=normed_components, probs=self.probs,)

    def denormalize(self, scale: Scale):
        denormed_components = [c.denormalize(scale) for c in self.components]
        return self.__class__(components=denormed_components, probs=self.probs,)

    # Structured

    @classmethod
    def structure(cls, params):
        (class_params, numeric_params) = params
        (mixture_class, component_classes) = class_params
        (mixture_params, component_params) = numeric_params
        (probs,) = mixture_params
        components = [
            c_classes[0].structure((c_classes, c_params))
            for (c_classes, c_params) in zip(component_classes, component_params)
        ]
        mixture_class = class_params[0]
        mixture = mixture_class(components=components, probs=probs,)
        return mixture

    def destructure(self):
        component_classes, component_numeric = zip(
            *[c.destructure() for c in self.components]
        )
        self_numeric = (self.probs,)
        class_params = (self.__class__, component_classes)
        numeric_params = (self_numeric, component_numeric)
        return (class_params, numeric_params)

    # Optimizable

    @classmethod
    def from_params(
        cls, fixed_params, opt_params, scale=None, traceable=True
    ):  # FIXME: traceable; why sometimes no Scale?
        if not scale:
            scale = Scale(0.0, 1.0)
        floor = fixed_params.get("floor", -np.inf)
        ceiling = fixed_params.get("ceiling", np.inf)
        # Allow logistic center to exceed the range by 20%
        loc_min = np.maximum(scale.low, floor) - 0.2 * scale.width
        loc_max = np.minimum(scale.high, ceiling) + 0.2 * scale.width
        loc_range = loc_max - loc_min
        structured_params = opt_params.reshape((-1, 3))
        locs = loc_min + scipy.special.expit(structured_params[:, 0]) * loc_range
        # Allow logistic scales between 0.01 and 0.5
        # Don't allow tiny scales outside of the visible range
        s_min = 0.01 + 0.1 * np.where(
            (locs < scale.low),
            scale.low - locs,
            np.where(locs > scale.high, locs - scale.high, 0.0),
        )
        s_max = 0.5
        s_range = s_max - s_min
        ss = s_min + scipy.special.expit(structured_params[:, 1]) * s_range
        # Allow probs > 0.01
        probs = list(
            0.01
            + nn.softmax(structured_params[:, 2])
            * (1 - 0.01 * structured_params[:, 2].size)
        )
        # Bundle up components
        component_logistics = [
            Logistic(l, s, scale, normalized=True) for (l, s) in zip(locs, ss)
        ]
        components = [
            Truncate(base_dist=cl, floor=floor, ceiling=ceiling)
            for cl in component_logistics
        ]
        mixture = cls(components=components, probs=probs)
        return mixture

    @staticmethod
    def initialize_optimizable_params(fixed_params):
        num_components = fixed_params["num_components"]
        loc_multiplier = 3
        s_multiplier = 1.5
        locs = (onp.random.rand(num_components) - 0.5) * loc_multiplier
        scales = (onp.random.rand(num_components) - 0.5) * s_multiplier
        weights = onp.random.rand(num_components)
        components = onp.stack([locs, scales, weights]).transpose()
        return components.reshape(-1)

    @classmethod
    def normalize_fixed_params(self, fixed_params, scale):
        normed_fixed_params = dict(fixed_params)
        normed_fixed_params["floor"] = scale.normalize_point(
            fixed_params.get("floor", -np.inf)
        )
        normed_fixed_params["ceiling"] = scale.normalize_point(
            fixed_params.get("ceiling", np.inf)
        )
        return normed_fixed_params

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
