import jax.numpy as np

from . import condition


class SmoothnessCondition(condition.Condition):
    def loss(self, dist) -> float:
        return self.weight * (
            np.sum(np.square(np.diff(dist.normed_log_densities, n=2))) / 2
            + np.sum(np.square(np.diff(dist.normed_log_densities, n=1))) / 100
        )

    def destructure(self):
        return ((SmoothnessCondition,), (self.weight,))

    def __str__(self):
        return "Minimize rough edges in the distribution"

    def __repr__(self):
        return f"SmoothnessCondition(weight={self.weight})"
