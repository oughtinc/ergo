from ergo.distributions import point_density
import ergo.static as static

from . import condition


class WassersteinCondition(condition.Condition):
    p_dist: "point_density.PointDensity"
    weight: float = 1.0

    def __init__(self, p_dist, weight=1.0):
        self.p_dist = p_dist
        super().__init__(weight)

    def loss(self, q_dist) -> float:
        return self.weight * static.wasserstein_distance(
            self.p_dist.normed_densities, q_dist.normed_densities
        )

    def destructure(self):
        dist_classes, dist_numeric = self.p_dist.destructure()
        cond_numeric = (self.weight,)
        return ((WassersteinCondition, dist_classes), (cond_numeric, dist_numeric))

    @classmethod
    def structure(cls, params):
        class_params, numeric_params = params
        cond_class, dist_classes = class_params
        cond_numeric, dist_numeric = numeric_params
        dist_params = (dist_classes, dist_numeric)
        dist = dist_classes[0].structure(dist_params)
        return cls(dist, cond_numeric[0])

    def __str__(self):
        return "Minimize the Wasserstein distance between the two distributions"
