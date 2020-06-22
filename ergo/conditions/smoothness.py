import jax.numpy as np

from ergo.utils import shift

from . import condition


class SmoothnessCondition(condition.Condition):
    def loss(self, dist) -> float:
        window_size = 5
        squared_distance = 0.0
        for i in range(1, window_size + 1):
            squared_distance += (1 / i ** 2) * np.sum(
                np.square(
                    dist.normed_log_densities
                    - shift(dist.normed_log_densities, i, dist.normed_log_densities[0])
                )
            )
        return self.weight * squared_distance / dist.normed_log_densities.size

    def destructure(self):
        return ((SmoothnessCondition,), (self.weight,))

    def __str__(self):
        return "Minimize rough edges in the distribution"
