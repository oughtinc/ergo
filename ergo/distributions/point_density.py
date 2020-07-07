from dataclasses import dataclass

from jax import nn
import jax.numpy as np
import numpy as onp
from scipy.interpolate import interp1d

from ergo.scale import Scale

from . import constants
from .distribution import Distribution
from .optimizable import Optimizable

# from ergo.utils import trapz


@dataclass
class PointDensity(Distribution, Optimizable):
    """
    A distribution specified through a number of density points.
    """

    normed_xs: np.DeviceArray
    normed_densities: np.DeviceArray
    bin_probs: np.DeviceArray
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

        self.scale = scale
        init_np = np if traceable else onp

        xs = init_np.array(xs)
        densities = init_np.array(densities)

        if normalized:
            self.normed_xs = xs
            self.normed_densities = densities

        else:
            self.normed_xs = scale.normalize_points(xs)
            self.normed_densities = scale.normalize_densities(self.normed_xs, densities)
        # auc =  trapz(self.normed_densities, x=self.normed_xs)
        # print(auc)

        self.bin_sizes = init_np.full(self.normed_xs.size, 1 / self.normed_xs.size)
        self.bin_probs = self.normed_densities * self.bin_sizes
        # print(f"sum of bin probs is: {np.sum(self.bin_probs)}")
        self.grid = init_np.linspace(0, 1, self.normed_xs.size + 1)

        if cumulative_normed_ps is not None:
            self.cumulative_normed_ps = cumulative_normed_ps

        else:
            self.cumulative_normed_ps = np.append(
                np.array([0]),
                init_np.cumsum(self.bin_probs)
            )
        self.normed_log_densities = init_np.log(self.normed_densities)

    # Distribution

    def pdf(self, x):
        """
        If x is out of distribution range, returns 0. Otherwise returns the
        density at the lowest bin for which the lower bound of the bin
        is greater than or equal to x.

        :param x: The point in the distribution to get the density at
        """

        x = self.scale.normalize_point(x)
        bin = np.argmin(np.abs(self.normed_xs - x))
        # bin = np.where(x > self.normed_xs[-1], -1, np.argmax(self.normed_xs >= x))
        return np.where(
            (x < 0) | (x > 1),
            0,
            self.scale.denormalize_density(
                self.scale.denormalize_point(self.normed_xs[bin]),
                self.normed_densities[bin],
            ),
        )

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
            return self.scale.denormalize_density(
                self.scale.denormalize_point(normed_x), normed_density
            )

        def out_of_range_pdf(normed_x):
            return np.where(
                normed_x == self.normed_xs[0],
                self.scale.denormalize_density(
                    self.scale.denormalize_point(self.normed_xs[0]),
                    self.normed_densities[0],
                ),
                np.where(
                    normed_x == self.normed_xs[-1],
                    self.scale.denormalize_density(
                        self.scale.denormalize_point(self.normed_xs[-1]),
                        self.normed_densities[-1],
                    ),
                    0,
                ),
            )

        return np.where(
            (normed_x <= self.normed_xs[0]) | (normed_x >= self.normed_xs[-1]),
            out_of_range_pdf(normed_x),
            in_range_pdf(normed_x),
        )
        """

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def cdf(self, x):
        '''
        x = self.scale.normalize_point(x)
        # bin = np.where(x > self.normed_xs[-1], -1, np.argmax(self.normed_xs >= x))
        bin = np.argmin(np.abs(self.normed_xs - x))
        return np.where(x < 0, 0, np.where(x > 1, 1, self.cumulative_normed_ps[bin]))
        '''

        normed_x = self.scale.normalize_point(x)

        def in_range_cdf(normed_x):
            bin = np.argmax(self.grid > normed_x) - 1
            c_below_bin = np.where(bin > 0, self.cumulative_normed_ps[bin], 0)
            c_in_bin = self.normed_densities[bin] * (normed_x - self.grid[bin])
            return c_below_bin + c_in_bin

        return np.where(
            normed_x <= self.grid[0],
            0,
            np.where(normed_x >= self.grid[-1], 1, in_range_cdf(normed_x)),
        )

    def ppf(self, q):
        bin = np.argmin(np.abs(self.cumulative_normed_ps - q))
        return self.scale.denormalize_point(self.grid[bin])

        """
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
        """

    def sample(self):
        raise NotImplementedError

    # Scaling

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

    # Create

    @classmethod
    def from_pairs(
        cls, pairs, scale: Scale, normalized=False, allow_non_standard_pairs=True
    ):
        sorted_pairs = sorted([(v["x"], v["density"]) for v in pairs])
        xs = np.array([x for (x, density) in sorted_pairs])
        densities = np.array([density for (x, density) in sorted_pairs])
        if not normalized:
            xs = scale.normalize_points(xs)
            densities = scale.normalize_densities(xs, densities)

        grid = onp.linspace(0, 1, constants.point_density_default_num_points)
        target_xs = (grid[1:] + grid[:-1]) / 2

        if allow_non_standard_pairs:
            # interpolate ps at target_xs
            if not (
                len(xs) == len(target_xs)
                and np.isclose(xs, target_xs, rtol=1e-04).all()
            ):
                f = interp1d(xs, densities)
                densities = f(target_xs)

        # Make sure AUC is 1
        auc = np.sum(densities) / densities.size
        densities /= auc

        return cls(target_xs, densities, scale=scale, normalized=True)

    @classmethod
    def from_conditions(
        cls,
        *args,
        fixed_params=None,
        scale=None,
        num_points=constants.point_density_default_num_points,
        **kwargs,
    ):
        if scale is None:
            # TODO: Should we do this?
            scale = Scale(0, 1)

        grid = np.linspace(0, 1, num_points)
        target_xs = (grid[1:] + grid[:-1]) / 2

        if fixed_params is None:
            # TODO: Seems weird to denormalize when will be normalized in normalize_fixed_params
            fixed_params = {"xs": scale.denormalize_points(target_xs)}
        else:
            raise Exception("Point Density does not accept custom xs specifications")
            # fixed_params = {"xs": scale.denormalize_points(target_xs)}
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
        # ps = np.abs(opt_params)
        ps = nn.softmax(opt_params) * opt_params.size
        densities = ps
        return cls(
            xs=xs, densities=densities, scale=scale, normalized=True, traceable=True
        )

    # Optimize Create Helpers

    @staticmethod
    def initialize_optimizable_params(fixed_params):
        num_points = fixed_params["xs"].size
        return onp.full(num_points, 1.0)

    @classmethod
    def normalize_fixed_params(self, fixed_params, scale):
        return {"xs": scale.normalize_points(fixed_params["xs"])}

    # Export

    def to_arrays(self, metaculus_denorm=False):
        xs = self.scale.denormalize_points(self.normed_xs)

        if metaculus_denorm:
            # Make sure points cover whole scale, return normed densities
            densities = self.normed_densities
            if xs[0] != self.scale.low:
                density = (densities[0] - densities[1]) / 2 + densities[0]
                clamped_density = onp.maximum(density, 0)

                xs = onp.append(onp.array([self.scale.low]), xs)
                densities = onp.append(np.array([clamped_density]), densities)
            if xs[-1] != self.scale.high:
                density = (densities[-1] - densities[-2]) / 2 + densities[-1]
                clamped_density = onp.maximum(density, 0)

                xs = onp.append(xs, onp.array([self.scale.high]))
                densities = onp.append(densities, np.array([clamped_density]))
        else:
            densities = self.scale.denormalize_densities(xs, self.normed_densities)

        return xs, densities

    def to_pairs(self, metaculus_denorm=False):
        xs, densities = self.to_arrays(metaculus_denorm)
        pairs = [
            {"x": float(x), "density": float(density)}
            for x, density in zip(xs, densities)
        ]
        return pairs

    # Condition Methods

    def entropy(self):
        return -np.dot(self.bin_probs, np.log(self.bin_probs))

    def cross_entropy(self, q_dist):
        # We assume that the distributions are on the same scale!
        return -np.dot(self.bin_probs, np.log(q_dist.bin_probs))

    def mean(self):
        normed_mean = np.dot(self.normed_xs, self.bin_probs)
        return self.scale.denormalize_point(normed_mean)

    def variance(self):
        normed_mean = np.dot(self.normed_xs, self.bin_probs)
        normed_variance = np.dot(
            self.bin_probs, np.square(self.normed_xs - normed_mean)
        )
        return self.scale.denormalize_variance(normed_variance)
