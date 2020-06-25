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

    def normalize(self, scale: Scale):
        # TODO: Should we really just returns self here? Because the pdf and returned density are
        # both denormalized
        return self
        '''
        normalized_xs = scale.normalize_points(self.xs)
        normalized_densities = scale.normalize_densities(self.xs, self.densities)
        return self.__class__(normalized_xs, normalized_densities, self.weight)
        '''

    def denormalize(self, scale: Scale):
        return self
        '''
        denormalized_xs = scale.denormalize_points(self.xs)
        denormalized_densities = scale.denormalize_densities(self.xs, self.densities)
        return self.__class__(denormalized_xs, denormalized_densities, self.weight)
        '''

    def destructure(self):
        return ((HistogramCondition,), (self.xs, self.densities, self.weight))

    def __key(self):
        return (
            HistogramCondition,
            (tuple(self.xs), tuple(self.densities), self.weight),
        )

    def _describe_fit(self, dist):
        print(f'cond xs: {self.xs} cond densities: {self.densities}')
        description = super()._describe_fit(dist)

        def entry_distance_fn(x, density):
            return abs(1.0 - density / dist.pdf(x))

        distances = vmap(entry_distance_fn)(self.xs, self.densities)
        description["max_distance"] = np.max(distances)
        description["90th_distance"] = np.percentile(distances, 90)
        description["mean_distance"] = np.mean(distances)
        return description

    def __str__(self):
        return "The probability density function looks similar to the provided density function."
