"""
Mixture distributions

This module contains both the base Mixture Class as well as the Location-Scale Family
Mixture Subclass.
"""
from dataclasses import dataclass, field
from functools import partial
import itertools
from typing import List, Optional, Sequence, Type, TypeVar

from jax import grad, jit, nn, scipy
import jax.numpy as np
import numpy as onp
import scipy as oscipy

from ergo.utils import minimize

from .base import categorical
from .conditions import Condition
from .distribution import Distribution
from .location_scale_family import LSDistribution
from .scale import Scale

M = TypeVar("M", bound="Mixture")


@dataclass
class Mixture(Distribution):
    components: Sequence[Distribution]
    probs: List[float]

    def __mul__(self, x):
        return self.__class__(
            [component * x for component in self.components], self.probs
        )

    def cdf(self, x):
        return np.sum([c.cdf(x) * p for c, p in zip(self.components, self.probs)])

    def ppf(self, q):
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
            return self.components[0].ppf(q)
        ppfs = [c.ppf(q) for c in self.components]
        cmin = np.min(ppfs)
        cmax = np.max(ppfs)
        try:
            return oscipy.optimize.bisect(
                lambda x: self.cdf(x) - q,
                cmin - abs(cmin / 100),
                cmax + abs(cmax / 100),
                maxiter=1000,
            )
        except ValueError:
            return (cmax + cmin) / 2

    def sample(self):
        i = categorical(np.array(self.probs))
        component_dist = self.components[i]
        return component_dist.sample()

    @staticmethod
    def initialize_params(num_components):
        raise NotImplementedError("This should be implemented by a subclass")

    def to_params(self):
        # This is like destructure, but from_params transforms probs using softmax,
        # so x != from_params(to_params(x))
        raise NotImplementedError("This should be implemented by a subclass")

    def denormalize(self, scale_min, scale_max):
        """
        Assume that the distribution has been normalized to be over [0,1].
        Return the distribution on the true scale of [scale_min, scale_max]

        :param scale_min: the true-scale minimum of the range
        :param scale_max: the true-scale maximum of the range
        """
        raise NotImplementedError("This should be implemented by a subclass")

    def normalize(self, scale_min, scale_max):
        """
        Assume that the distribution's true range is [scale_min, scale_max].
        Return the normalized condition.

        :param scale_min: the true-scale minimum of the range
        :param scale_max: the true-scale maximum of the range
        :return: the condition normalized to [0,1]
        """
        raise NotImplementedError("This should be implemented by a subclass")

    @staticmethod
    def loss_jac(clas, scale_min, scale_max, conditions):
        normalized_conditions = [
            condition.normalize(scale_min, scale_max) for condition in conditions
        ]

        cond_data = [condition.destructure() for condition in normalized_conditions]
        if cond_data:
            cond_classes, cond_params = zip(*cond_data)
        else:
            cond_classes, cond_params = [], []

        loss = lambda params: static_loss(  # noqa: E731
            clas, params, cond_classes, cond_params
        )
        jac = lambda params: static_loss_grad(  # noqa: E731
            clas, params, cond_classes, cond_params
        )

        return loss, jac

    @classmethod
    def from_samples(cls, data, num_components=3, verbose=False) -> M:
        data = np.array(data)
        scale = Scale(scale_min=min(data), scale_max=max(data))
        normalized_data = np.array([scale.normalize_point(datum) for datum in data])

        # FIXME (#219): This is pretty inefficient

        @jit
        def loss(params):
            dist = cls.from_params(params)
            normed_params = dist.to_params()
            return -cls.params_logpdf(normed_params, normalized_data)

        @jit
        def jac(params):
            dist = cls.from_params(params)
            normed_params = dist.to_params()
            return -cls.params_gradlogpdf(normed_params, normalized_data)

        normalized_mixture: M = cls.from_loss(loss, jac, num_components, verbose)
        return normalized_mixture.denormalize(scale.scale_min, scale.scale_max)

    @classmethod
    def from_params(cls, params):
        raise NotImplementedError("This should be implemented by a subclass")

    @classmethod
    def from_conditions(
        cls,
        conditions: Sequence[Condition],
        num_components: Optional[int] = None,
        verbose=False,
        scale_min=0,
        scale_max=1,
        init_tries=100,
        opt_tries=10,
    ) -> M:
        """
        Fit a mixture distribution from Conditions

        :param conditions: conditions to fit
        :param num_components: number of components to include in the mixture.
        :param init_tries:
        :param opt_tries:
        :param verbose:
        :param scale_min: the true-scale minimum of the range to fit over.
        :param scale_max: the true-scale maximum of the range to fit over.
        :return: the fitted mixture
        """
        loss, jac = cls.loss_jac(cls, scale_min, scale_max, conditions)

        normalized_mixture: M = cls.from_loss(
            loss=loss,
            jac=jac,
            num_components=num_components,
            verbose=verbose,
            init_tries=init_tries,
            opt_tries=opt_tries,
        )
        return normalized_mixture.denormalize(scale_min, scale_max)

    @classmethod
    def from_loss(
        cls,
        loss,
        jac,
        num_components: Optional[int] = None,
        verbose=False,
        init_tries=100,
        opt_tries=10,
    ) -> M:
        onp.random.seed(0)

        init = lambda: cls.initialize_params(num_components)  # noqa: E731

        fit_results = minimize(
            loss,
            init=init,
            jac=jac,
            init_tries=init_tries,
            opt_tries=opt_tries,
            verbose=verbose,
        )
        if not fit_results.success and verbose:
            print(fit_results)
        final_params = fit_results.x

        return cls.from_params(final_params)

    def logpdf(self, data):
        return self.params_logpdf(self.to_params(), data)

    def logpdf1(self, datum):
        return self.params_logpdf1(self.to_params(), datum)

    def pdf1(self, datum):
        # Not calling logpdf1 because we only want to call
        # to_params once even if we call this with a vector
        return np.exp(self.params_logpdf1(self.to_params(), datum))

    @staticmethod
    def params_cdf(params, x):
        raise NotImplementedError

    @staticmethod
    def params_ppf(params, p):
        raise NotImplementedError


@dataclass
class LSMixture(Mixture):
    components: Sequence[LSDistribution]
    probs: List[float]
    component_type: Type[LSDistribution] = field(repr=False)

    @staticmethod
    def initialize_params(num_components, scale_multiplier=0.2):
        """
        Each component has (location, scale, weight).
        The shape of the components matrix is (num_components, 3).
        Weights sum to 1 (are given in log space).
        We use original numpy to initialize parameters since we don't
        want to track randomness.
        """
        locs = onp.random.rand(num_components)
        scales = onp.random.rand(num_components) * scale_multiplier
        weights = onp.full(num_components, -num_components)
        components = onp.stack([locs, scales, weights]).transpose()
        return components.reshape(-1)

    @classmethod
    def from_params(cls, params):
        structured_params = params.reshape((-1, 3))
        unnormalized_weights = structured_params[:, 2]
        probs = list(nn.softmax(unnormalized_weights))
        component_dists = [cls.component_type(p[0], p[1]) for p in structured_params]
        return cls(component_dists, probs)

    def to_params(self):
        nested_params = [
            [c.loc, c.scale, weight] for c, weight in zip(self.components, self.probs)
        ]
        return np.array(list(itertools.chain.from_iterable(nested_params)))

    def normalize(self, scale_min: float, scale_max: float):
        normalized_components = [
            component.normalize(scale_min, scale_max) for component in self.components
        ]
        return self.__class__(normalized_components, self.probs, self.component_type)

    def denormalize(self, scale_min: float, scale_max: float):
        denormalized_components = [
            component.denormalize(scale_min, scale_max) for component in self.components
        ]
        return self.__class__(denormalized_components, self.probs, self.component_type)


@partial(jit, static_argnums=(0, 2))
def static_loss(dist_class, dist_params, cond_classes, cond_params):
    print(
        f"Tracing {dist_class.__name__} loss for {[c.__name__ for c in cond_classes]}"
    )
    dist = dist_class.from_params(dist_params)
    total_loss = 0.0
    for (cond_class, cond_param) in zip(cond_classes, cond_params):
        condition = cond_class.structure(cond_param)
        total_loss += condition.loss(dist)
    return total_loss * 100


static_loss_grad = jit(grad(static_loss, argnums=1), static_argnums=(0, 2))
