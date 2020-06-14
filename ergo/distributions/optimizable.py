from abc import ABC, abstractmethod
from typing import Sequence, Type, TypeVar

import jax.numpy as np
import numpy as onp

from ergo.conditions import Condition
from ergo.scale import Scale
import ergo.static as static
from ergo.utils import minimize

T = TypeVar("T", bound="Optimizable")


class Optimizable(ABC):
    @classmethod
    @abstractmethod
    def from_params(cls, fixed_params, opt_params, traceable=True):
        ...

    @staticmethod
    @abstractmethod
    def initialize_optimizable_params(fixed_params):
        ...

    @abstractmethod
    def normalize(self):
        ...

    @abstractmethod
    def denormalize(self, scale: Scale):
        ...

    @classmethod
    def from_samples(
        cls: Type[T],
        data,
        fixed_params=None,
        scale=None,
        verbose=False,
        init_tries=1,
        opt_tries=1,
    ) -> T:
        if fixed_params is None:
            fixed_params = {}
        data = np.array(data)
        if scale is None:
            data_range = max(data) - min(data)
            scale = Scale(
                low=min(data) - 0.25 * data_range, high=max(data) + 0.25 * data_range,
            )

        fixed_params = cls.normalize_fixed_params(fixed_params, scale)
        normalized_data = np.array(scale.normalize_points(data))

        loss = lambda params: static.dist_logloss(  # noqa: E731
            cls, fixed_params, params, normalized_data
        )
        jac = lambda params: static.dist_grad_logloss(  # noqa: E731
            cls, fixed_params, params, normalized_data
        )

        normalized_dist = cls.from_loss(
            loss,
            jac,
            fixed_params=fixed_params,
            verbose=verbose,
            init_tries=init_tries,
            opt_tries=opt_tries,
        )

        return normalized_dist.denormalize(scale)

    @classmethod
    def from_conditions(
        cls: Type[T],
        conditions: Sequence[Condition],
        fixed_params=None,
        scale=None,
        verbose=False,
        init_tries=1,
        opt_tries=1,
        jit_all=False,
    ) -> T:
        if fixed_params is None:
            fixed_params = {}

        if scale is None:
            scale = Scale(0, 1)  # assume a linear scale in [0,1]

        fixed_params = cls.normalize_fixed_params(fixed_params, scale)

        normalized_conditions = [condition.normalize(scale) for condition in conditions]

        cond_data = [condition.destructure() for condition in normalized_conditions]
        if cond_data:
            cond_classes, cond_params = zip(*cond_data)
        else:
            cond_classes, cond_params = [], []

        if jit_all:
            jitted_loss = static.jitted_condition_loss
            jitted_jac = static.jitted_condition_loss_grad
        else:
            jitted_loss = static.condition_loss
            jitted_jac = static.condition_loss_grad

        def loss(opt_params):
            return jitted_loss(cls, fixed_params, opt_params, cond_classes, cond_params)

        def jac(opt_params):
            return jitted_jac(cls, fixed_params, opt_params, cond_classes, cond_params)

        normalized_dist = cls.from_loss(
            fixed_params=fixed_params,
            loss=loss,
            jac=jac,
            verbose=verbose,
            init_tries=init_tries,
            opt_tries=opt_tries,
        )

        return normalized_dist.denormalize(scale)

    @classmethod
    def from_loss(
        cls: Type[T],
        loss,
        jac,
        fixed_params=None,
        verbose=False,
        init_tries=1,
        opt_tries=1,
    ) -> T:

        # fixed_params are assumed to be normalized

        if fixed_params is None:
            fixed_params = {}

        onp.random.seed(0)

        init = lambda: cls.initialize_optimizable_params(fixed_params)  # noqa: E731

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
        optimized_params = fit_results.x

        return cls.from_params(fixed_params, optimized_params)

    @classmethod
    def normalize_fixed_params(self, fixed_params, scale):
        # They are not normalized unless a child class implements this method
        return fixed_params
