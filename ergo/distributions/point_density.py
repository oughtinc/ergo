from dataclasses import dataclass

import jax.nn as nn
import jax.numpy as np
import numpy as onp

from ergo.scale import Scale
from ergo.utils import trapz

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

        self.scale = scale.copy()
        init_np = np if traceable else onp

        xs = np.array(xs)
        densities = np.array(densities)

        auc = trapz(densities, x=xs)
        densities /= auc

        if normalized:
            self.normed_xs = xs
            self.normed_densities = densities

        else:
            self.normed_xs = scale.normalize_points(xs)
            self.normed_densities = scale.normalize_densities(self.normed_xs, densities)

        if cumulative_normed_ps is not None:
            self.cumulative_normed_ps = cumulative_normed_ps
        else:
            bin_probs = self.normed_bin_probs(
                self.normed_xs, self.normed_densities, numpy=init_np
            )
            self.cumulative_normed_ps = init_np.append(
                init_np.cumsum(bin_probs), init_np.array([1.0])
            )

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
            low_idx = np.argmax(self.normed_xs > normed_x) - 1
            high_idx = low_idx + 1
            low_density = self.normed_densities[low_idx]
            high_density = self.normed_densities[high_idx]
            low_x = self.normed_xs[low_idx]
            high_x = self.normed_xs[high_idx]
            dist = high_x - low_x
            normed_density = (normed_x - low_x) / dist * high_density + (
                high_x - normed_x
            ) / dist * low_density
            return self.scale.denormalize_density(self.scale.denormalize_point(normed_x), normed_density)

        def out_of_range_pdf(normed_x):
            return np.where(
                normed_x == self.normed_xs[0],
                self.scale.denormalize_density(self.scale.denormalize_point(self.normed_xs[0]), self.normed_densities[0]),
                np.where(
                    normed_x == self.normed_xs[-1],
                    self.scale.denormalize_density(self.scale.denormalize_point(self.normed_xs[-1]), self.normed_densities[-1]),
                    0,
                ),
            )

        return np.where(
            (normed_x <= self.normed_xs[0]) | (normed_x >= self.normed_xs[-1]),
            out_of_range_pdf(normed_x),
            in_range_pdf(normed_x),
        )

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def cdf(self, x):
        normed_x = self.scale.normalize_point(x)
        normed_x_density = self.scale.normalize_density(normed_x, self.pdf(x))

        def in_range_cdf(normed_x):
            bin = np.where(
                normed_x < self.normed_xs[-1],
                np.argmax(self.normed_xs > normed_x) - 1,
                self.normed_xs.size - 1,
            )
            c_below_bin = np.where(bin > 0, self.cumulative_normed_ps[bin - 1], 0)
            c_in_bin = (
                (normed_x_density + self.normed_densities[bin])
                / 2.0
                * (normed_x - self.normed_xs[bin])
            )
            return c_below_bin + c_in_bin

        return np.where(
            normed_x < self.normed_xs[0],
            0,
            np.where(normed_x > self.normed_xs[-1], 1, in_range_cdf(normed_x)),
        )

    def ppf(self, q):
        low_idx = np.argmax(self.cumulative_normed_ps >= q)
        high_idx = low_idx + 1
        low_x = self.normed_xs[low_idx]
        high_x = self.normed_xs[np.minimum(high_idx, self.normed_xs.size - 1)]
        low_cum = np.where(low_idx == 0, 0, self.cumulative_normed_ps[low_idx - 1])
        high_cum = self.cumulative_normed_ps[high_idx - 1]
        dist = high_cum - low_cum
        normed_x = np.where(
            dist == 0,
            low_x,
            (q - low_cum) / dist * high_x + (high_cum - q) / dist * low_x,
        )
        # TODO: change to denomrlize_points maybe?
        return self.scale.denormalize_point(normed_x)

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
                "xs": scale.denormalize_points(
                    np.linspace(0, 1, constants.point_density_default_num_points)
                )
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
        densities = ps 
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
        # print(f"This is the sum of the densities {sum(densities)}")
        return cls(xs, densities, scale=scale, normalized=normalized)

    def to_lists(self, metaculus_denorm=False):
        xs = self.scale.denormalize_points(self.normed_xs)

        densities = self.normed_densities if metaculus_denorm else self.scale.denormalize_densities(xs, self.normed_densities)
        return xs, densities

    def to_pairs(self, metaculus_denorm=False):
        xs, densities = self.to_lists(metaculus_denorm)
        pairs = [
            {"x": float(x), "density": float(density)}
            for x, density in zip(xs, densities)
        ]
        return pairs

    def to_arrays(self, metaculus_denorm=False):
        xs, densities = self.to_lists(metaculus_denorm)
        return xs, densities

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

    def mean(self):
        bin_sizes = np.diff(self.normed_xs)
        bin_xs = (self.normed_xs[1:] + self.normed_xs[:-1]) / 2
        bin_probs = (
            (self.normed_densities[1:] + self.normed_densities[:-1]) / 2.0 * bin_sizes
        )
        normed_mean = np.dot(bin_xs, bin_probs)
        return self.scale.denormalize_point(normed_mean)

    def variance(self):
        bin_sizes = np.diff(self.normed_xs)
        bin_xs = (self.normed_xs[1:] + self.normed_xs[:-1]) / 2
        bin_probs = (
            (self.normed_densities[1:] + self.normed_densities[:-1]) / 2.0 * bin_sizes
        )
        normed_mean = np.dot(bin_xs, bin_probs)
        normed_variance = np.dot(bin_probs, np.square(bin_xs - normed_mean))
        return self.scale.denormalize_variance(normed_variance)
