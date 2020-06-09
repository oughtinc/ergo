from jax import vmap
import jax.numpy as np

from ergo.scale import Scale

from . import condition


class HistogramCondition(condition.Condition):
    """
    The distribution should fit the specified histogram as closely as
    possible
    """

    xs: np.DeviceArray
    densities: np.DeviceArray
    weight: float = 1.0

    def __init__(self, xs, densities, weight=1.0):
        self.xs = xs
        self.densities = densities
        super().__init__(weight)

    def loss(self, dist):
        entry_loss_fn = lambda x, density: (density - dist.pdf(x)) ** 2  # noqa: E731
        total_loss = np.sum(vmap(entry_loss_fn)(self.xs, self.densities))
        return self.weight * total_loss / self.xs.size

    def normalize(self, scale_min: float, scale_max: float):
        scale = Scale(scale_min, scale_max)
        normalized_xs = np.array([scale.normalize_point(x) for x in self.xs])
        normalized_densities = np.array(
            [density * scale.range for density in self.densities]
        )
        return self.__class__(normalized_xs, normalized_densities, self.weight)

    def denormalize(self, scale_min: float, scale_max: float):
        scale = Scale(scale_min, scale_max)
        denormalized_xs = np.array([scale.denormalize_point(x) for x in self.xs])
        denormalized_densities = np.array(
            [density / scale.range for density in self.densities]
        )
        return self.__class__(denormalized_xs, denormalized_densities, self.weight)

    def destructure(self):
        return (HistogramCondition, (self.xs, self.densities, self.weight))

    def __key(self):
        return (
            HistogramCondition,
            (tuple(self.xs), tuple(self.densities), self.weight),
        )

    def _describe_fit(self, dist):
        description = super()._describe_fit(dist)
        entry_distance_fn = lambda x, density: abs(density - dist.pdf(x))  # noqa: E731
        distances = vmap(entry_distance_fn)(self.xs, self.densities)
        description["max_distance"] = np.max(distances)
        description["95th_distance"] = np.percentile(distances, 90)
        description["mean_distance"] = np.mean(distances)
        return description

    def __str__(self):
        return "The probability density function looks similar to the provided density function."
