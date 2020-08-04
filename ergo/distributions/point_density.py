from dataclasses import dataclass

from backports.cached_property import cached_property
from jax import nn
import jax.numpy as np
import numpy as onp
from scipy.interpolate import interp1d

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

        self.scale = scale

        if normalized:
            self.normed_xs = xs
            self.normed_densities = densities

        else:
            self.normed_xs = scale.normalize_points(xs)
            self.normed_densities = scale.normalize_densities(self.normed_xs, densities)

        self._cumulative_normed_ps = cumulative_normed_ps

    @cached_property
    def bin_probs(self):
        return self.normed_densities * constants.bin_sizes

    @cached_property
    def normed_log_densities(self):
        return np.log(self.normed_densities)

    @property
    def cumulative_normed_ps(self):
        if self._cumulative_normed_ps is None:
            self._cumulative_normed_ps = np.append(
                np.array([0]), np.cumsum(self.bin_probs)
            )
        return self._cumulative_normed_ps

    @cached_property
    def true_xs(self):
        return self.scale.denormalize_points(self.normed_xs)

    @cached_property
    def true_grid(self):
        return self.scale.denormalize_points(constants.grid)

    # Distribution

    def pdf(self, x):
        """
        If x is out of distribution range, returns 0. Otherwise,
        returns the true ("denormalized") density at the point in
        self.normed_xs which is closest to normalized x by absolute
        difference.

        :param x: The point at which to get the probability density
        """
        x = self.scale.normalize_point(x)
        bin = np.argmin(np.abs(self.normed_xs - x))
        return np.where(
            (x < 0) | (x > 1),
            0,
            self.scale.denormalize_density(
                self.scale.denormalize_point(self.normed_xs[bin]),
                self.normed_densities[bin],
            ),
        )

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def cdf(self, x):
        """
        If x is below the distribution range, returns 0. If x is
        above the distribution range, returns 1.
        Otherwise, it returns the closest bin edge and returns the cdf to that point

        :param x: The point at which to get the cumulative density
        """

        x = self.scale.normalize_point(x)
        bin = np.argmin(np.abs(constants.grid - x))
        return np.where(x < 0, 0, np.where(x > 1, 1, self.cumulative_normed_ps[bin]))

    def ppf(self, q):
        bin = np.argmin(np.abs(self.cumulative_normed_ps - q))
        return self.true_grid[bin]

    def sample(self):
        raise NotImplementedError

    # Scaling

    def normalize(self):
        return PointDensity(
            self.normed_xs,
            self.normed_densities,
            scale=Scale(0.0, 1.0),
            normalized=True,
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
    def from_pairs(cls, pairs, scale: Scale, normalized=False, interpolate=True):
        sorted_pairs = sorted([(v["x"], v["density"]) for v in pairs])
        xs = np.array([x for (x, density) in sorted_pairs])
        densities = np.array([density for (x, density) in sorted_pairs])

        if not normalized:
            xs = scale.normalize_points(xs)
            densities = scale.normalize_densities(xs, densities)

        if interpolate:
            # interpolate ps at target_xs
            if not (
                len(xs) == len(constants.target_xs)
                and np.isclose(xs, constants.target_xs, rtol=1e-04).all()
            ):
                f = interp1d(xs, densities)
                densities = f(constants.target_xs)

        # Make sure AUC is 1
        auc = np.sum(densities) / densities.size
        densities /= auc

        return cls(constants.target_xs, densities, scale=scale, normalized=True)

    @classmethod
    def from_conditions(
        cls, *args, fixed_params=None, scale=None, **kwargs,
    ):

        if fixed_params is None:
            # TODO: Seems weird to denormalize when will be normalized in normalize_fixed_params
            fixed_params = {"xs": scale.denormalize_points(constants.target_xs)}
        else:
            raise Exception("Point Density does not accept custom xs specifications")

        return super(PointDensity, cls).from_conditions(
            *args, fixed_params=fixed_params, scale=scale, **kwargs
        )

    @classmethod
    def from_params(cls, fixed_params, opt_params, scale=None, traceable=True):
        if scale is None:
            scale = Scale(0.0, 1.0)
        xs = fixed_params["xs"]

        densities = nn.softmax(opt_params) * opt_params.size
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

    @classmethod
    def add_endpoints(cls, xs, densities, scale):
        """
        Returns a list of xs and densities with endpoints that are on the edge of the scale
        provided. If no scale is provided assume data is normalized.
        """

        if xs[0] != scale.low:
            xdiff_ratio = (xs[1] - xs[0]) / xs[0]
            density = (densities[0] - densities[1]) / xdiff_ratio + densities[0]
            clamped_density = onp.maximum(density, 0)

            xs = onp.append(onp.array([scale.low]), xs)
            densities = onp.append(np.array([clamped_density]), densities)

        if xs[-1] != scale.high:
            xdiff_ratio = (xs[-1] - xs[-2]) / (scale.high - xs[-1])
            density = (densities[-1] - densities[-2]) / xdiff_ratio + densities[-1]
            clamped_density = onp.maximum(density, 0)

            xs = onp.append(xs, onp.array([scale.high]))
            densities = onp.append(densities, np.array([clamped_density]))

        return xs, densities

    def to_arrays(self, denorm_xs_only=False, add_endpoints=False, num_xs=None):
        """
        Exports the distribution in two arrays of xs and associated densities

        :param denorm_xs_only: only denormalize the xs and leave the densities normalized
        :param add_endpoints: add endpoints of the distribution to points passed out
        :param num_xs: the number of points to summarise the distribution with
        :return: the true-scaled x values and densities (the normed density if denorm_xs_only=True)
        """

        if num_xs is not None:
            grid = np.linspace(0, 1, num_xs + 1)
            normed_xs = (grid[1:] + grid[:-1]) / 2
            xs = self.scale.denormalize_points(normed_xs)
            f = interp1d(self.true_xs, self.normed_densities)
            normed_densities = f(xs)
        else:
            xs = self.true_xs
            normed_densities = self.normed_densities

        if add_endpoints:
            xs, normed_densities = PointDensity.add_endpoints(
                xs, normed_densities, self.scale
            )

        if denorm_xs_only:
            densities = normed_densities
        else:
            densities = self.scale.denormalize_densities(xs, normed_densities)

        return xs, densities

    def to_pairs(self, *args, **kwargs):
        xs, densities = self.to_arrays(*args, **kwargs)
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
        return np.dot(self.true_xs, self.bin_probs)

    def variance(self):
        mean = self.mean()
        return np.dot(self.bin_probs, np.square(self.true_xs - mean))
