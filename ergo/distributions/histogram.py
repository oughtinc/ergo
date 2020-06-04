from dataclasses import dataclass
from functools import partial
from typing import List

from jax import grad, jit, nn
import jax.numpy as np
import numpy as onp
import scipy as oscipy

from . import conditions, distribution, scale


@dataclass
class HistogramDist(distribution.Distribution):
    logps: np.DeviceArray

    def __init__(
        self, logps=None, scale_min=0, scale_max=1, traceable=False, direct_init=None
    ):
        if direct_init:
            self.logps = direct_init["logps"]
            self.ps = direct_init["ps"]
            self.cum_ps = direct_init["cum_ps"]
            self.bins = direct_init["bins"]
            self.size = direct_init["size"]
            self.scale_min = direct_init["scale_min"]
            self.scale_max = direct_init["scale_max"]
        else:
            init_numpy = np if traceable else onp
            self.logps = logps
            self.ps = np.exp(logps)
            self.cum_ps = np.array(init_numpy.cumsum(self.ps))
            self.bins = np.linspace(scale_min, scale_max, logps.size + 1)
            self.size = logps.size
            self.scale_min = scale_min
            self.scale_max = scale_max
        self.scale = scale.Scale(scale_min, scale_max)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, conditions.Condition):
            return self.__key() == other.__key()
        return NotImplemented

    def __key(self):
        return tuple(self.logps)

    def entropy(self):
        return -np.dot(self.ps, self.logps)

    def cross_entropy(self, q_dist):
        # Uncommented to support Jax tracing:
        # assert self.scale_min == q_dist.scale_min, (self.scale_min, q_dist.scale_min)
        # assert self.scale_max == q_dist.scale_max
        # assert self.size == q_dist.size, (self.size, q_dist.size)
        return -np.dot(self.ps, q_dist.logps)

    def pdf(self, x):
        return self.ps[np.argmax(self.bins >= self.scale.normalize_point(x))]

    def cdf(self, x):
        return self.cum_ps[np.argmax(self.bins >= self.scale.normalize_point(x))]

    def ppf(self, q):
        return self.scale.denormalize_point(
            np.where(self.cum_ps >= q)[0][0] / self.cum_ps.size
        )

    def sample(self):
        raise NotImplementedError

    def rv(self):
        raise NotImplementedError

    def normalize(self, true_scale: scale.Scale = None):
        """
        Normalize the histogram onto [0,1]

        Setting a true_scale allows you to express that the histogram to be normalized
        did not cover the entire scale of interest.

        E.g. -- imagine my histogram only has probability mass from 2 to 5,
        but I'm interested in p over [0,10].
        So I'll use Scale(0,10), and I'll get back a histogram with additional
        bins with 0 probability to cover the area from 0 to 2 and 5 to 10,
        where my histogram doesn't place any probability.

        :param true_scale: the full scale that I'm interested in probability over.
        """
        # if true_scale is not provided, assume that the histogram has
        # entries over the entire scale of interest
        if not true_scale:
            return HistogramDist(self.logps, 0, 1)

        if true_scale.scale_min is None or true_scale.scale_max is None:
            raise ValueError(
                "If you provide a true_scale, you must provide both a min and max"
            )

        if (
            true_scale.scale_min > self.scale_min
            or true_scale.scale_max < self.scale_max
        ):
            raise ValueError(
                "Can only rescale hist to a scale that includes all of its current scale"
            )

        x_range_below = scale.Scale(true_scale.scale_min, self.scale_min)
        x_range_below_per_hist_range = x_range_below.range / self.scale.range
        num_x_bins_below = round(self.size * x_range_below_per_hist_range)

        x_range_above = scale.Scale(self.scale_max, true_scale.scale_max)
        x_range_above_per_hist_range = x_range_above.range / self.scale.range
        num_x_bins_above = round(self.size * x_range_above_per_hist_range)

        bins_below = onp.full(num_x_bins_below, float("-inf"))

        bins_above = onp.full(num_x_bins_above, float("-inf"))

        logps = onp.concatenate((bins_below, self.logps, bins_above))

        return HistogramDist(logps, 0, 1)

    def denormalize(self, scale_min, scale_max):
        return HistogramDist(self.logps, scale_min, scale_max)

    @classmethod
    def from_conditions(
        cls,
        conditions: List["conditions.Condition"],
        scale_min=0,
        scale_max=1,
        num_bins=100,
        verbose=False,
    ):
        normalized_conditions = [
            condition.normalize(scale_min, scale_max) for condition in conditions
        ]

        cond_data = [condition.destructure() for condition in normalized_conditions]
        if cond_data:
            cond_classes, cond_params = zip(*cond_data)
        else:
            cond_classes, cond_params = [], []

        loss = lambda params: static_loss(  # noqa: E731
            params, cond_classes, cond_params
        )
        jac = lambda params: static_loss_grad(  # noqa: E731
            params, cond_classes, cond_params
        )

        normalized_dist = cls.from_loss(loss=loss, jac=jac, num_bins=num_bins)

        if verbose:
            for condition in normalized_conditions:
                print(condition)
                print(condition.describe_fit(normalized_dist))

        return normalized_dist.denormalize(scale_min, scale_max)

    @classmethod
    def from_loss(cls, loss, jac, num_bins=100):
        x0 = cls.initialize_params(num_bins)
        results = oscipy.optimize.minimize(loss, jac=jac, x0=x0)
        return cls.from_params(results.x)

    @classmethod
    def from_params(cls, params, traceable=False):
        logps = nn.log_softmax(params)
        return cls(logps, traceable=traceable)

    def destructure(self):
        return (
            HistogramDist,
            (
                self.logps,
                self.ps,
                self.cum_ps,
                self.bins,
                self.size,
                self.scale_min,
                self.scale_max,
            ),
        )

    @classmethod
    def structure(cls, params):
        return cls(
            direct_init={
                "logps": params[0],
                "ps": params[1],
                "cum_ps": params[2],
                "bins": params[3],
                "size": params[4],
                "scale_min": params[5],
                "scale_max": params[6],
            }
        )

    @classmethod
    def from_pairs(cls, pairs):
        sorted_pairs = sorted([(v["x"], v["density"]) for v in pairs])
        xs = [x for (x, density) in sorted_pairs]
        densities = [density for (x, density) in sorted_pairs]
        scale_min = xs[0]
        scale_max = xs[-1]
        logps = onp.log(onp.array(densities) / sum(densities))
        return cls(logps, scale_min=scale_min, scale_max=scale_max)

    def to_pairs(self):
        pairs = []
        bins = onp.array(self.bins)
        ps = onp.array(self.ps)
        for i, bin in enumerate(bins[:-1]):
            x = float((bin + bins[i + 1]) / 2.0)
            bin_size = float(bins[i + 1] - bin)
            density = float(ps[i]) / bin_size
            pairs.append({"x": x, "density": density})
        return pairs

    def to_lists(self):
        xs = []
        densities = []
        bins = onp.array(self.bins)
        ps = onp.array(self.ps)
        for i, bin in enumerate(bins[:-1]):
            x = float((bin + bins[i + 1]) / 2.0)
            bin_size = float(bins[i + 1] - bin)
            density = float(ps[i]) / bin_size
            xs.append(x)
            densities.append(density)
        return xs, densities

    def to_arrays(self):
        # TODO: vectorize
        xs, densities = self.to_lists()
        return np.array(xs), np.array(densities)

    @staticmethod
    def initialize_params(num_bins):
        return onp.full(num_bins, -num_bins)


def static_loss(dist_params, cond_classes, cond_params):
    total_loss = 0.0
    for (cond_class, cond_param) in zip(cond_classes, cond_params):
        total_loss += static_condition_loss(dist_params, cond_class, cond_param)
    return total_loss


def static_loss_grad(dist_params, cond_classes, cond_params):
    total_grad = 0.0
    for (cond_class, cond_param) in zip(cond_classes, cond_params):
        total_grad += static_condition_loss_grad(dist_params, cond_class, cond_param)
    return total_grad


@partial(jit, static_argnums=1)
def static_condition_loss(dist_params, cond_class, cond_param):
    print(f"Tracing condition loss for {cond_class.__name__} with params {cond_param}")
    dist = HistogramDist.from_params(dist_params, traceable=True)
    condition = cond_class.structure(cond_param)
    return condition.loss(dist) * 100


static_condition_loss_grad = jit(
    grad(static_condition_loss, argnums=0), static_argnums=1
)
