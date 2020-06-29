from dataclasses import asdict, dataclass, field
from datetime import timedelta
import time
from typing import TypeVar

from jax import grad, vmap
import jax.numpy as np


@dataclass
class Scale:
    low: float
    high: float
    width: float = field(init=False)

    def __post_init__(self):
        self.width = self.high - self.low

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Scale):
            return self.__key() == other.__key()
        return NotImplemented

    def __key(self):
        cls, params = self.destructure()
        return (cls, params)

    def normalize_point(self, point):
        return (point - self.low) / self.width

    def denormalize_point(self, point):
        return (point * self.width) + self.low

    def denormalize_points(self, points):
        return np.array([self.denormalize_point(point) for point in points])

    def normalize_points(self, points):
        return np.array([self.normalize_point(point) for point in points])

    def normalize_variance(self, variance):
        if variance is None:
            raise Exception("Point was None This shouldn't happen")
        return variance / (self.width ** 2)

    def denormalize_variance(self, variance):
        if variance is None:
            raise Exception("Point was None This shouldn't happen")
        return variance * (self.width ** 2)

    def normalize_density(self, _, density):
        return density * self.width

    def denormalize_density(self, _, density):
        return density / self.width

    def normalize_densities(self, _, densities):
        return densities * self.width

    def denormalize_densities(self, _, densities):
        return densities / self.width

    def destructure(self):
        return ((Scale,), (self.low, self.high))

    @classmethod
    def structure(cls, params):
        classes, numeric = params
        return classes[0](*numeric)

    def export(self):
        export_dict = asdict(self)
        export_dict["class"] = type(self).__name__
        return export_dict


ScaleClass = TypeVar("ScaleClass", bound=Scale)


@dataclass
class LogScale(Scale):
    log_base: float

    def __post_init__(self):
        # if self.log_base < 1:
        #     raise ValueError(f"log_Base must be > 1, was {self.log_base}")
        self.width = self.high - self.low
        self.density_denorm = grad(self.normalize_point)
        self.density_norm = grad(self.denormalize_point)

    def __hash__(self):
        return super().__hash__()

    # def normalize_density(self, original_x, density):
    #     normed_x = self.normalize_point(original_x)
    #     normed_xbar = normed_x + 0.001
    #     original_xbar = self.denormalize_point(normed_xbar)
    #     density_ratio = (original_xbar - original_x) / (normed_xbar - normed_x)
    #     return density * density_ratio

    # def denormalize_density(self, normed_x, density):
    #     original_x = self.denormalize_point(normed_x)
    #     normed_xbar = normed_x + 0.001
    #     original_xbar = self.denormalize_point(normed_xbar)
    #     density_ratio = (normed_xbar - normed_x) / (original_xbar - original_x)
    #     return density * density_ratio

    def normalize_density(self, normed_x, density):
        return density * self.density_norm(normed_x)

    def denormalize_density(self, true_x, density):
        return density * self.density_denorm(true_x)

    def normalize_densities(self, normed_xs, densities):
        return densities * vmap(self.density_norm)(normed_xs)

    def denormalize_densities(self, true_xs, densities):
        return densities * vmap(self.density_denorm)(true_xs)

    def normalize_point(self, point):
        """
        Get a prediction sample value on the normalized scale from a true-scale value

        :param true_value: a sample value on the true scale
        :return: a sample value on the normalized scale
        """
        if point is None:
            raise Exception("Point was None This shouldn't happen")

        shifted = point - self.low
        numerator = shifted * (self.log_base - 1)
        scaled = numerator / self.width
        timber = 1 + scaled
        floored_timber = np.maximum(timber, 1e-9)

        return np.log(floored_timber) / np.log(self.log_base)

    def denormalize_point(self, point):
        """
        Get a value on the true scale from a normalized-scale value

        :param normalized_value: [description]
        :type normalized_value: [type]
        :return: [description]
        :rtype: [type]
        """
        if point is None:
            raise Exception("Point was None This shouldn't happen")

        deriv_term = (self.log_base ** point - 1) / (self.log_base - 1)
        scaled = self.width * deriv_term
        return self.low + scaled

    def destructure(self):
        return ((LogScale,), (self.low, self.high, self.log_base))

    @classmethod
    def structure(cls, params):
        classes, numeric = params
        low, high, log_base = numeric
        return cls(low, high, log_base)


@dataclass
class TimeScale(Scale):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.width = self.high - self.low

    def __repr__(self):
        return (
            f"TimeScale(low={self.timestamp_to_str(self.low)}, "
            f"high={self.timestamp_to_str(self.high)}, "
            f"width={timedelta(seconds=self.width)})"
        )

    def __hash__(self):
        return super().__hash__()

    def destructure(self):
        return (
            (TimeScale,),
            (self.low, self.high,),
        )

    def timestamp_to_str(self, timestamp: float) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(timestamp))


def scale_factory(scale_dict):
    scale_class = scale_dict["class"]
    low = scale_dict["low"]
    high = scale_dict["high"]

    if scale_class == "Scale":
        return Scale(low, high)
    if scale_class == "LogScale":
        return LogScale(low, high, scale_dict["log_base"])
    if scale_class == "TimeScale":
        return TimeScale(low, high)
    raise NotImplementedError(
        f"reconstructing scales of class {scale_class} is not implemented."
    )
