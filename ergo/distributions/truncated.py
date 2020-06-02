from dataclasses import dataclass
from .distribution import Distribution
from .scale import Scale
from typing import List, Optional, Sequence, Type, TypeVar
from .mixture import static_loss, static_loss_grad
import numpy as onp
from .conditions import Condition
from ergo.utils import minimize


def truncate(underlying_dist_class: Distribution, floor: float, ceiling: float):
    @dataclass
    class TruncatedDist(Distribution):
        underlying_dist: Distribution

        def rv(self):
            raise NotImplementedError

        def sample(self):
            raise NotImplementedError

        def cdf(self, x):
            raise NotImplementedError

        def ppf(self, q):
            raise NotImplementedError

        def pdf1(self, x):
            # jit needs this as purely functional code
            # use np.where
            # if x < floor or x > ceiling:
            #     return 0

            p_below = self.underlying_dist.cdf(floor)
            p_above = 1 - self.underlying_dist.cdf(ceiling)
            p_inside = 1 - (p_below + p_above)

            return self.underlying_dist.pdf1(x) * (1 / p_inside)

        @classmethod
        def from_params(cls, params):
            underlying_dist = underlying_dist_class.from_params(params)
            return cls(underlying_dist)

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
        ):
            normalized_conditions = [
                condition.normalize(scale_min, scale_max) for condition in conditions
            ]

            cond_data = [condition.destructure() for condition in normalized_conditions]
            if cond_data:
                cond_classes, cond_params = zip(*cond_data)
            else:
                cond_classes, cond_params = [], []

            loss = lambda params: static_loss(  # noqa: E731
                cls, params, cond_classes, cond_params
            )
            jac = lambda params: static_loss_grad(  # noqa: E731
                cls, params, cond_classes, cond_params
            )

            normalized_dist = underlying_dist_class.from_loss(  # type: ignore
                loss=loss,
                jac=jac,
                num_components=num_components,
                verbose=verbose,
                init_tries=init_tries,
                opt_tries=opt_tries,
            )
            return normalized_dist.denormalize(scale_min, scale_max)

        # @classmethod
        # def from_loss(
        #     cls,
        #     underlying_dist,
        #     loss,
        #     jac,
        #     num_components: Optional[int] = None,
        #     verbose=False,
        #     init_tries=100,
        #     opt_tries=10,
        # ):
        #     onp.random.seed(0)

        #     init = lambda: underlying_dist.initialize_params(num_components)  # noqa: E731

        #     fit_results = minimize(
        #         loss,
        #         init=init,
        #         jac=jac,
        #         init_tries=init_tries,
        #         opt_tries=opt_tries,
        #         verbose=verbose,
        #     )
        #     if not fit_results.success and verbose:
        #         print(fit_results)
        #     final_params = fit_results.x

        #     return underlying_dist.from_params(final_params)

    return TruncatedDist
