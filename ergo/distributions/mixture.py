from dataclasses import dataclass
from typing import Sequence

from jax import scipy
import jax.numpy as np
import scipy as oscipy

from ergo.scale import Scale

from .base import categorical
from .distribution import Distribution


@dataclass
class Mixture(Distribution):
    components: Sequence[Distribution]
    probs: Sequence[float]

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        raise NotImplementedError

    def grad_logpdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        # TODO: vectorize
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

    def normalize(self, scale: Scale):
        """
        Return the distribution on the true scale

        :param scale: the true-scale of condition
        :return: the condition in the true scale
        """
        normalized_components = [
            component.normalize(scale) for component in self.components
        ]
        return self.__class__(normalized_components, self.probs)

    def denormalize(self, scale: Scale):
        """
        Return the normalized condition.

        :param scale: the true-scale of condition
        :return: the condition normalized to [0,1]
        """
        denormalized_components = [
            component.denormalize(scale) for component in self.components
        ]
        return self.__class__(denormalized_components, self.probs)
