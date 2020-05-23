"""
Mixture distributions

This module contains both the base Mixture Class as well as the Location-Scale Family
Mixture Subclass.
"""
from dataclasses import dataclass, field
import itertools
from typing import List, Optional, Sequence, Type, TypeVar
import warnings

from jax import grad, jit, nn, scipy
import jax.numpy as np
import numpy as onp
import scipy as oscipy

from ergo.utils import minimize

from .base import categorical
from .conditions import Condition, IntervalCondition
from .distribution import Distribution
from .location_scale_family import LSDistribution

M = TypeVar("M", bound="Mixture")


@dataclass
class Mixture(Distribution):
    components: Sequence[Distribution]
    probs: List[float]

    def __mul__(self, x):
        return self.__class__(
            [component * x for component in self.components], self.probs
        )

    def rv(self,):
        raise NotImplementedError("No access to mixture rv at this time")

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
        return oscipy.optimize.bisect(
            lambda x: self.cdf(x) - q,
            np.min(ppfs) - 0.01,
            np.max(ppfs) + 0.01,
            maxiter=1000,
        )

    def sample(self):
        i = categorical(np.array(self.probs))
        component_dist = self.components[i]
        return component_dist.sample()

    @staticmethod
    def initialize_params(num_components):
        raise NotImplementedError("This should be implemented by a subclass")

    def to_params(self):
        raise NotImplementedError("This should be implemented by a subclass")

    def get_denormalized(self, scale_min, scale_max):
        """
        Assume that the distribution has been normalized to be over [0,1].
        Return the distribution on the true scale of [scale_min, scale_max]

        :param scale_min: the true-scale minimum of the range
        :param scale_max: the true-scale maximum of the range
        """
        raise NotImplementedError("This should be implemented by a subclass")

    def get_normalized(self, scale_min, scale_max):
        """
        Assume that the distribution's true range is [scale_min, scale_max].
        Return the normalized condition.

        :param scale_min: the true-scale minimum of the range
        :param scale_max: the true-scale maximum of the range
        :return: the condition normalized to [0,1]
        """
        raise NotImplementedError("This should be implemented by a subclass")

    def to_conditions(self, verbose=False):
        """
        Convert mixture to a set of percentile statements that
        determines the mixture.
        """
        warnings.warn(
            "to_conditions is a proof-of-concept for finding parameterized"
            " conditions using optimization. It's not ready to be used!"
        )

        def condition_from_params(params):
            percentile = nn.softmax(np.array([params[0], 1]))[0]
            return IntervalCondition(p=percentile, value=params[1])

        def loss(params):
            condition = condition_from_params(params)
            return condition.loss(self)

        jac = grad(loss)
        init_params = np.array([0.1, 0.1])  # percentile, value
        fit_results = minimize(loss, x0=init_params, jac=jac, tries=5, verbose=verbose)
        if not fit_results.success and verbose:
            print(fit_results)
        final_params = fit_results.x
        return condition_from_params(final_params)

    @classmethod
    def from_samples(
        cls, data, initial_dist: Optional[M] = None, num_components=3, verbose=False,
    ) -> M:
        data = np.array(data)
        z = float(np.mean(data))
        normalized_data = data / z

        def loss(params):
            return -cls.params_logpdf(params, normalized_data)

        def jac(params):
            return -cls.params_gradlogpdf(params, normalized_data)

        dist = cls.from_loss(loss, jac, initial_dist, num_components, verbose)
        return dist * z

    @classmethod
    def from_params(cls, params):
        raise NotImplementedError("This should be implemented by a subclass")

    @classmethod
    def from_conditions(
        cls,
        conditions: List[Condition],
        initial_dist: Optional[M] = None,
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
        :param initial_dist: mixture distribution to start from
        (should be normalizes, i.e. on [0,1]).
        Takes precedence over num_components
        :param num_components: number of components to include in the mixture.
        initial_dist take precedence
        :param tries:
        :param scale_min: the true-scale minimum of the range to fit over.
        :param scale_max: the true-scale maximum of the range to fit over.
        :return: the fitted mixture
        """
        normalized_conditions = [
            condition.get_normalized(scale_min, scale_max) for condition in conditions
        ]

        def loss(params):
            dist = cls.from_params(params)
            total_loss = sum(
                condition.loss(dist) for condition in normalized_conditions
            )
            return total_loss * 100

        loss = jit(loss)
        jac = jit(grad(loss))
        normalized_mixture = cls.from_loss(
            loss=loss,
            jac=jac,
            initial_dist=initial_dist,
            num_components=num_components,
            verbose=verbose,
            init_tries=init_tries,
            opt_tries=opt_tries,
        )
        return normalized_mixture.get_denormalized(scale_min, scale_max)

    @classmethod
    def from_loss(
        cls,
        loss,
        jac,
        initial_dist: Optional[M] = None,
        num_components: Optional[int] = None,
        verbose=False,
        init_tries=100,
        opt_tries=10,
    ) -> M:
        onp.random.seed(0)

        if initial_dist:
            init = lambda: initial_dist.to_params()  # noqa: E731
        elif num_components:
            init = lambda: cls.initialize_params(num_components)  # noqa: E731
        else:
            raise ValueError("Need to provide either num_components or initial_dist")

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

    def get_normalized(self, scale_min: float, scale_max: float):
        normalized_components = [
            component.get_normalized(scale_min, scale_max)
            for component in self.components
        ]
        return self.__class__(normalized_components, self.probs, self.component_type)

    def get_denormalized(self, scale_min: float, scale_max: float):
        denormalized_components = [
            component.get_denormalized(scale_min, scale_max)
            for component in self.components
        ]
        return self.__class__(denormalized_components, self.probs, self.component_type)
