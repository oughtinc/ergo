"""
Mixture of logistic distributions
"""
from dataclasses import dataclass
import itertools
from typing import Optional, Sequence

from jax import nn
import jax.numpy as np
import numpy as onp

from ergo.conditions import Condition
from ergo.scale import Scale
import ergo.static as static
from ergo.utils import minimize

from .logistic import Logistic
from .mixture import Mixture


@dataclass
class LogisticMixture(Mixture):
    components: Sequence[Logistic]
    probs: Sequence[float]

    def logpdf(self, x):
        return static.logistic_mixture_logpdf(self.to_params(), x)

    def grad_logpdf(self, x):
        return static.logistic_mixture_grad_logpdf(self.to_params(), x)

    def to_params(self):
        nested_params = [
            [c.loc, c.scale, weight] for c, weight in zip(self.components, self.probs)
        ]
        return np.array(list(itertools.chain.from_iterable(nested_params)))

    @classmethod
    def from_params(cls, params):
        structured_params = params.reshape((-1, 3))
        unnormalized_weights = structured_params[:, 2]
        probs = list(nn.softmax(unnormalized_weights))
        component_dists = [Logistic(p[0], p[1]) for p in structured_params]
        return cls(component_dists, probs)

    @classmethod
    def from_samples(cls, data, num_components=3, verbose=False) -> "LogisticMixture":
        data = np.array(data)
        scale = Scale(scale_min=min(data), scale_max=max(data))
        normalized_data = np.array([scale.normalize_point(datum) for datum in data])

        loss = lambda params: static.dist_logloss(  # noqa: E731
            cls, params, normalized_data
        )
        jac = lambda params: static.dist_grad_logloss(  # noqa: E731
            cls, params, normalized_data
        )

        normalized_mixture = cls.from_loss(loss, jac, num_components, verbose)
        return normalized_mixture.denormalize(scale.scale_min, scale.scale_max)

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
    ) -> "LogisticMixture":
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
        normalized_conditions = [
            condition.normalize(scale_min, scale_max) for condition in conditions
        ]

        cond_data = [condition.destructure() for condition in normalized_conditions]
        if cond_data:
            cond_classes, cond_params = zip(*cond_data)
        else:
            cond_classes, cond_params = [], []

        loss = lambda params: static.jitted_condition_loss(  # noqa: E731
            cls, params, cond_classes, cond_params
        )
        jac = lambda params: static.jitted_condition_loss_grad(  # noqa: E731
            cls, params, cond_classes, cond_params
        )

        normalized_mixture: LogisticMixture = cls.from_loss(
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
    ) -> "LogisticMixture":
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
