from dataclasses import dataclass

import jax.numpy as np

from ergo.scale import Scale

from .distribution import Distribution


@dataclass
class Truncate(Distribution):
    base_dist: Distribution
    floor: float = -np.inf  # true scale
    ceiling: float = np.inf  # true scale

    def __post_init__(self):
        # https://github.com/tensorflow/probability/blob/master/discussion/where-nan.pdf
        self.p_below = np.where(
            self.floor == -np.inf,
            0,
            self.base_dist.cdf(np.where(self.floor == -np.inf, 0, self.floor)),
        )
        self.p_above = np.where(
            self.ceiling == np.inf,
            0,
            1.0 - self.base_dist.cdf(np.where(self.ceiling == np.inf, 1, self.ceiling)),
        )
        self.p_inside = 1.0 - (self.p_below + self.p_above)

    # Distribution

    def pdf(self, x):
        p_x = np.exp(self.base_dist.logpdf(x) - np.log(self.p_inside))
        return np.where(x < self.floor, 0.0, np.where(x > self.ceiling, 0.0, p_x),)

    def logpdf(self, x):
        logp_x = self.base_dist.logpdf(x) - np.log(self.p_inside)
        return np.where(
            x < self.floor, -np.inf, np.where(x > self.ceiling, -np.inf, logp_x)
        )

    def cdf(self, x):
        c_x = (self.base_dist.cdf(x) - self.p_below) / self.p_inside
        return np.where(x < self.floor, 0.0, np.where(x > self.ceiling, 1.0, c_x))

    def ppf(self, q):
        """
        Percent point function (inverse of cdf) at q.
        """
        return self.base_dist.ppf(self.p_below + q * self.p_inside)

    def sample(self):
        success = False
        while not success:
            s = self.base_dist.sample()
            if s > self.floor and s < self.ceiling:
                success = True
        return s

    # Scaled

    @property
    def scale(self):
        return self.base_dist.scale

    def normalize(self):
        normed_base_dist = self.base_dist.normalize()
        normed_floor = (
            self.scale.normalize_point(self.floor)
            if self.floor != -np.inf
            else self.floor
        )
        normed_ceiling = (
            self.scale.normalize_point(self.ceiling)
            if self.ceiling != np.inf
            else self.ceiling
        )
        return self.__class__(
            base_dist=normed_base_dist, floor=normed_floor, ceiling=normed_ceiling,
        )

    def denormalize(self, scale: Scale):
        denormed_base_dist = self.base_dist.denormalize(scale)
        denormed_floor = (
            scale.denormalize_point(self.floor, as_string=False)
            if self.floor != -np.inf
            else self.floor
        )
        denormed_ceiling = (
            scale.denormalize_point(self.ceiling, as_string=False)
            if self.ceiling != np.inf
            else self.ceiling
        )
        return self.__class__(
            base_dist=denormed_base_dist,
            floor=denormed_floor,
            ceiling=denormed_ceiling,
        )

    # Structured

    @classmethod
    def structure(self, params):
        class_params, numeric_params = params
        (self_class, base_classes) = class_params
        (self_numeric, base_numeric) = numeric_params
        base_dist = base_classes[0].structure((base_classes, base_numeric))
        return self_class(
            base_dist=base_dist, floor=self_numeric[0], ceiling=self_numeric[1],
        )

    def destructure(self):
        self_class, self_numeric = (
            self.__class__,
            (self.floor, self.ceiling),
        )
        base_classes, base_numeric = self.base_dist.destructure()
        class_params = (self_class, base_classes)
        numeric_params = (self_numeric, base_numeric)
        return (class_params, numeric_params)
