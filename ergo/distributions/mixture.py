"""
Mixture distributions

Jax jitting and scipy optimization don't handle classes well so we'll
partially have to work with arrays directly (all the params_*
classmethods).
"""
from dataclasses import dataclass
from typing import Any, List, Optional, TypeVar
import warnings

from jax import grad, jit, nn, scipy
import jax.numpy as np
import scipy as oscipy

from ergo.utils import minimize

from .base import categorical
from .conditions import Condition, PercentileCondition
from .distribution import Distribution

# from .Logistic import Logistic
# from .Normal import Normal
M = TypeVar("M", bound="Mixture")


@dataclass
class Mixture(Distribution):
    components: List[
        Any
    ]  # Once from_samples() is refactored into singular distributions  Union[List[Logistic], List[Normal]]
    probs: List[float]

    def __mul__(self, x):
        return self.__class__(
            [component * x for component in self.components], self.probs
        )

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
            lambda x: self.cdf(x) - q, np.min(ppfs), np.max(ppfs),
        )

    def cdf(self, x):
        return np.sum([c.cdf(x) * p for c, p in zip(self.components, self.probs)])

    def sample(self):
        i = categorical(np.array(self.probs))
        component_dist = self.components[i]
        return component_dist.sample()

    @staticmethod
    def initialize_params(num_components):
        raise NotImplementedError("This should be implemented by a subclass")

    def to_params(self):
        raise NotImplementedError("This should be implemented by a subclass")

    def to_percentiles(self, percentiles=None):
        if percentiles is None:
            percentiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        values = [self.ppf(q) for q in percentiles]
        return [
            PercentileCondition(percentile, value)
            for (percentile, value) in zip(percentiles, values)
        ]

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
            return PercentileCondition(percentile=percentile, value=params[1])

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
    ) -> M:
        def _loss(params):
            dist = cls.from_params(params)
            total_loss = 0.0
            for condition in conditions:
                total_loss += condition.loss(dist)
            return total_loss * 100

        loss = jit(_loss)
        jac = jit(grad(loss))

        return cls.from_loss(loss, jac, initial_dist, num_components, verbose)

    @classmethod
    def from_loss(
        cls,
        loss,
        jac,
        initial_dist: Optional[M] = None,
        num_components: Optional[int] = None,
        verbose=False,
    ) -> M:
        if initial_dist:
            init = lambda: initial_dist.to_params()  # noqa: E731
        elif num_components:
            init = lambda: cls.initialize_params(num_components)  # noqa: E731
        else:
            raise ValueError("Need to provide either num_components or initial_dist")

        fit_results = minimize(loss, init=init, jac=jac, tries=5, verbose=verbose)
        if not fit_results.success and verbose:
            print(fit_results)
        final_params = fit_results.x

        return cls.from_params(final_params)

    def logpdf(self, data):
        return self.params_logpdf(self.to_params(), data)

    def logpdf1(self, datum):
        return self.params_logpdf1(self.to_params(), datum)

    @staticmethod
    def params_cdf(params, x):
        raise NotImplementedError

    @staticmethod
    def params_ppf(params, p):
        raise NotImplementedError
