"""
Mixture distributions

Jax jitting and scipy optimization don't handle classes well so we'll
partially have to work with arrays directly (all the params_*
classmethods).
"""
from dataclasses import dataclass
from functools import partial
from typing import List, Union

from jax import grad, jit, scipy, vmap
import jax.numpy as np
import scipy as oscipy

from .base import categorical
from .conditions import PercentileCondition
from .distribution import Distribution
from .logistic import Logistic
from .normal import Normal


@dataclass
class Mixture(Distribution):
    components: Union[List[Logistic], List[Normal]]
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

    def to_percentiles(self, percentiles=None):
        if percentiles is None:
            percentiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        values = [self.ppf(q) for q in percentiles]
        return [
            PercentileCondition(percentile, value)
            for (percentile, value) in zip(percentiles, values)
        ]

    def sample(self):
        i = categorical(np.array(self.probs))
        component_dist = self.components[i]
        return component_dist.sample()

    def logpdf(self, data):
        return self.params_logpdf(self.to_params(), data)

    def logpdf1(self, datum):
        return self.params_logpdf1(self.to_params(), datum)

    @classmethod
    def params_logpdf(cls, params, data):
        return _mixture_params_logpdf(params, data)

    @classmethod
    def params_gradlogpdf(cls, params, data):
        return _mixture_params_gradlogpdf(params, data)

    @classmethod
    def params_cdf(cls, params, x):
        raise NotImplementedError

    @classmethod
    def params_ppf(cls, params, p):
        raise NotImplementedError


@jit
def _mixture_params_logpdf(cls, params, data):
    scores = vmap(partial(cls.params_logpdf1, params))(data)
    return np.sum(scores)


_mixture_params_gradlogpdf = jit(grad(_mixture_params_logpdf, argnums=0))
