from dataclasses import dataclass

import jax.numpy as np
import numpy as onp

from ergo.scale import Scale

from . import constants
from .distribution import Distribution
from .optimizable import Optimizable


@dataclass
class PointDensity(Distribution, Optimizable):
    """
    A distribution specified through a number of density points.
    """

    normed_xs: np.DeviceArray
    normed_densities: np.DeviceArray
    scale: Scale

    def __init__(
        self,
        xs,
        densities,
        scale: Scale,
        normalized=False,
        traceable=False,
        cumulative_normed_ps=None,
    ):
        if scale is None:
            raise ValueError
        init_np = np if traceable else onp
        if normalized:
            self.normed_xs = xs
            self.normed_densities = densities
        else:
            self.normed_xs = scale.normalize_points(xs)
            self.normed_densities = scale.normalize_densities(xs, densities)
        if cumulative_normed_ps is not None:
            self.cumulative_normed_ps = cumulative_normed_ps
        else:
            bin_probs = self.normed_bin_probs(
                self.normed_xs, self.normed_densities, numpy=init_np
            )
            self.cumulative_normed_ps = init_np.append(
                init_np.cumsum(bin_probs), init_np.array([1.0])
            )
        self.scale = scale
        self.normed_log_densities = np.log(self.normed_densities)

    # Distribution

    def pdf(self, x):
        """
        If x is out of distribution range, returns 0. Otherwise returns the
        density at the lowest bin for which the lower bound of the bin
        is greater than or equal to x.

        :param x: The point in the distribution to get the density at
        """
        normed_x = self.scale.normalize_point(x)

        def in_range_pdf(normed_x):
            low_idx = np.where(normed_x < self.normed_xs[-1], np.argmax(self.normed_xs > normed_x) - 1, self.normed_xs.size - 1)
            high_idx = np.minimum(low_idx + 1, self.normed_xs.size - 1)
            low_density = self.normed_densities[low_idx]
            high_density = self.normed_densities[high_idx]
            low_x = self.normed_xs[low_idx]
            high_x = self.normed_xs[high_idx]
            dist = high_x - low_x
            normed_density = np.where(dist == 0, low_density, (normed_x - low_x) / dist * high_density + (high_x - normed_x) / dist * low_density)
            return self.scale.denormalize_density(normed_x, normed_density)

        return np.where((normed_x < 0) | (normed_x > 1), 0, in_range_pdf(normed_x))

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def cdf(self, x):
        normed_x = self.scale.normalize_point(x)
        x_normed_density = self.scale.normalize_density(x, self.pdf(x))

        def in_range_cdf(normed_x):
            bin = np.where(normed_x < self.normed_xs[-1], np.argmax(self.normed_xs > normed_x) - 1, self.normed_xs.size - 1)
            c_below_bin = np.where(bin > 0, self.cumulative_normed_ps[bin-1], 0)
            c_in_bin = (x_normed_density + self.normed_densities[bin]) / 2.0 * (normed_x - self.normed_xs[bin])
            return c_below_bin + c_in_bin

        return np.where(normed_x < 0, 0, np.where(normed_x > 1, 1, in_range_cdf(normed_x)))

    def ppf(self, q):
        return self.scale.denormalize_point(
            self.normed_xs[np.argmax(self.cumulative_normed_ps >= q)]
        )

    def sample(self):
        raise NotImplementedError

    # Scaled

    def normalize(self):
        return PointDensity(
            self.normed_xs, self.normed_densities, scale=Scale(0, 1), normalized=True
        )

    def denormalize(self, scale: Scale):
        return PointDensity(
            self.normed_xs, self.normed_densities, scale=scale, normalized=True
        )

    # Structured

    @classmethod
    def structure(cls, params):
        class_params, numeric_params = params
        density_class, scale_classes = class_params
        density_numeric, scale_numeric = numeric_params
        return density_class(
            xs=density_numeric[0],
            densities=density_numeric[1],
            scale=scale_classes[0].structure((scale_classes, scale_numeric)),
            normalized=True,
            traceable=True,
            cumulative_normed_ps=density_numeric[2],
        )

    def destructure(self):
        scale_classes, scale_numeric = self.scale.destructure()
        class_params = (self.__class__, scale_classes)
        self_numeric = self.normed_xs, self.normed_densities, self.cumulative_normed_ps
        numeric_params = (self_numeric, scale_numeric)
        return (class_params, numeric_params)

    # Optimizable

    @classmethod
    def from_conditions(cls, *args, fixed_params=None, scale=None, **kwargs):
        if scale is None:
            # TODO: Should we do this?
            scale = Scale(0, 1)
        if fixed_params is None:
            # TODO: Seems weird to denormalize when will be normalized in normalize_fixed_params
            fixed_params = {
                "xs": scale.denormalize_points(np.linspace(
                    0, 1, constants.point_density_default_num_points
                ))
            }
        return super(PointDensity, cls).from_conditions(
            *args, fixed_params=fixed_params, scale=scale, **kwargs
        )

    @classmethod
    def from_params(cls, fixed_params, opt_params, scale=None, traceable=True):
        # TODO: traceable is always True here
        if scale is None:
            # TODO: Should we do this?
            scale = Scale(0, 1)
        xs = fixed_params["xs"]
        ps = np.abs(opt_params)
        Z = np.sum(cls.normed_bin_probs(xs, ps))
        densities = ps / Z
        # print(np.sum(cls.normed_bin_probs(xs, densities)))
        return cls(
            xs=xs, densities=densities, scale=scale, normalized=True, traceable=True
        )

    @staticmethod
    def initialize_optimizable_params(fixed_params):
        num_points = fixed_params["xs"].size
        return onp.full(num_points, 1.0)

    @classmethod
    def normalize_fixed_params(self, fixed_params, scale):
        return {"xs": scale.normalize_points(fixed_params["xs"])}

    # Other

    @classmethod
    def from_pairs(cls, pairs, scale: Scale, normalized=False):
        sorted_pairs = sorted([(v["x"], v["density"]) for v in pairs])
        xs = [x for (x, density) in sorted_pairs]
        densities = [density for (x, density) in sorted_pairs]
        return cls(xs, densities, scale=scale, normalized=normalized)

    def to_lists(self):
        xs = self.scale.denormalize_points(self.normed_xs)
        densities = self.scale.denormalize_densities(self.xs, self.normed_densities)
        return xs, densities

    def to_pairs(self):
        xs, densities = self.to_lists()
        pairs = [
            {"x": float(x), "density": float(density)}
            for x, density in zip(xs, densities)
        ]
        return pairs

    def to_arrays(self):
        xs, densities = self.to_lists()
        return np.array(xs), np.array(densities)

    def entropy(self):
        # We assume that the distributions are on the same scale!
        return -np.dot(self.normed_densities, self.normed_log_densities)

    def cross_entropy(self, q_dist):
        # We assume that the distributions are on the same scale!
        return -np.dot(self.normed_densities, q_dist.normed_log_densities)

    @classmethod
    def normed_bin_probs(cls, normed_xs, normed_densities, numpy=np):
        bin_sizes = numpy.diff(normed_xs)
        bin_probs = (normed_densities[1:] + normed_densities[:-1]) / 2.0 * bin_sizes
        return bin_probs
