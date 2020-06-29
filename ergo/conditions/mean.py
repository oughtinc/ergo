from ergo.scale import Scale

from . import condition


class MeanCondition(condition.Condition):
    """
    The distribution should have as close to the specified mean as possible.
    """

    mean: float
    weight: float = 1.0

    def __init__(self, mean, weight=1.0):
        self.mean = mean
        super().__init__(weight)

    def actual_mean(self, dist) -> float:
        # FIXME: Should be interacting with PointDensity via pdf
        #        or similar public interface
        """
        xs = np.linspace(
            dist.scale.low, dist.scale.high, dist.normed_densities.size
        )  # FIXME: Denormalize densities?
        return np.dot(dist.normed_densities, xs)
        """
        return dist.mean()

    def loss(self, dist) -> float:
        return self.weight * (self.actual_mean(dist) - self.mean) ** 2

    def _describe_fit(self, dist):
        description = super()._describe_fit(dist)
        description["mean"] = self.actual_mean(dist)
        return description

    def normalize(self, scale: Scale):
        normalized_mean = scale.normalize_point(self.mean)
        return self.__class__(normalized_mean, self.weight)

    def denormalize(self, scale: Scale):
        denormalized_mean = scale.denormalize_point(self.mean)
        return self.__class__(denormalized_mean, self.weight)

    def destructure(self):
        return ((MeanCondition,), (self.mean, self.weight))

    def __str__(self):
        return f"The mean is {self.mean}."
