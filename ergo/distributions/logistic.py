"""
Logistic distributions

Jax jitting and scipy optimization don't handle classes well so we'll
partially have to work with arrays directly (all the params_*
classmethods).
"""
from dataclasses import dataclass
import itertools
from typing import Any, Dict, List, Optional
import warnings

from jax import grad, jit, nn, scipy
import jax.numpy as np
import numpy as onp
import scipy as oscipy

from ergo.utils import minimize

from .conditions import Condition, PercentileCondition
from .distribution import Distribution
from .mixture import Mixture


@dataclass
class Logistic(Distribution):
    loc: float
    scale: float
    metadata: Optional[Dict[str, Any]]

    def __init__(self, loc: float, scale: float, metadata=None):
        self.loc = loc
        self.scale = np.max([scale, 0.0000001])  # Do not allow values <= 0
        self.metadata = metadata

    def __mul__(self, x):
        return Logistic(self.loc * x, self.scale * x)

    def rv(self,):
        return oscipy.stats.logistic(loc=self.loc, scale=self.scale)

    def sample(self):
        # FIXME (#296): This needs to be compatible with ergo sampling
        return onp.random.logistic(loc=self.loc, scale=self.scale)

    def cdf(self, x):
        y = (x - self.loc) / self.scale
        return scipy.stats.logistic.cdf(y)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf) at q.
        """
        return self.rv().ppf(q)

    @classmethod
    def from_samples_scipy(cls, samples) -> "Logistic":
        with onp.errstate(all="raise"):  # type: ignore
            loc, scale = oscipy.stats.logistic.fit(samples)
            return cls(loc, scale)

    @classmethod
    def from_conditions(
        self, conditions: List[Condition], initial_dist: Optional["Logistic"] = None
    ):
        raise NotImplementedError

    @staticmethod
    def from_samples(samples) -> "Logistic":
        mixture = LogisticMixture.from_samples(
            samples, num_components=1
        )  # TODO consider refactoring
        return mixture.components[0]

    @staticmethod
    @jit
    def params_logpdf(x, loc, scale):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logistic.html
        y = (x - loc) / scale
        return scipy.stats.logistic.logpdf(y) - np.log(scale)


@dataclass
class LogisticMixture(Mixture):
    components: List[Logistic]
    probs: List[float]

    def __mul__(self, x):
        return LogisticMixture(
            [component * x for component in self.components], self.probs
        )

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

    def to_params(self):
        nested_params = [
            [c.loc, c.scale, weight] for c, weight in zip(self.components, self.probs)
        ]
        return np.array(list(itertools.chain.from_iterable(nested_params)))

    @classmethod
    def from_params(cls, params):
        structured_params = params.reshape((-1, 3))
        unnormalized_weights = structured_params[:, 2]
        probs = list(np.exp(nn.log_softmax(unnormalized_weights)))
        component_dists = [Logistic(p[0], p[1]) for p in structured_params]
        return cls(component_dists, probs)

    @classmethod
    def from_samples(
        cls,
        data,
        initial_dist: Optional["LogisticMixture"] = None,
        num_components=3,
        verbose=False,
    ):
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
    def from_conditions(
        cls,
        conditions: List[Condition],
        initial_dist: Optional["LogisticMixture"] = None,
        num_components: Optional[int] = None,
        verbose=False,
    ):
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
        initial_dist: Optional["LogisticMixture"] = None,
        num_components: Optional[int] = None,
        verbose=False,
    ):
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
