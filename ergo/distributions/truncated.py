from dataclasses import dataclass
from .distribution import Distribution
import jax.numpy as np
from .conditions import Condition
from typing import Sequence, Optional


def truncate(underlying_dist_class: Distribution, floor: float, ceiling: float):
    @dataclass
    class TruncatedDist(Distribution):
        underlying_dist: Distribution

        def ppf(self, q):
            raise NotImplementedError

        def pdf1(self, x):
            p_below = self.underlying_dist.cdf(floor)
            p_above = 1 - self.underlying_dist.cdf(ceiling)
            p_inside = 1 - (p_below + p_above)

            p_at_x = self.underlying_dist.pdf1(x) * (1 / p_inside)

            return np.where(x < floor, 0, np.where(x > ceiling, 0, p_at_x))

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
            loss, jac = underlying_dist_class.loss_jac(
                cls, scale_min, scale_max, conditions
            )

            normalized_dist = underlying_dist_class.from_loss(  # type: ignore
                loss=loss,
                jac=jac,
                num_components=num_components,
                verbose=verbose,
                init_tries=init_tries,
                opt_tries=opt_tries,
            )
            return cls(normalized_dist.denormalize(scale_min, scale_max))

    return TruncatedDist
