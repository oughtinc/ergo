from dataclasses import asdict, dataclass, field
from datetime import timedelta
import time
from typing import TypeVar

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
        return [self.denormalize_point(point) for point in points]

    def normalize_points(self, points):
        return [self.normalize_point(point) for point in points]

    def normalize_variance(self, variance):
        if variance is None:
            raise Exception("Point was None This shouldn't happen")
        return variance / (self.width ** 2)

    def denormalize_variance(self, variance):
        if variance is None:
            raise Exception("Point was None This shouldn't happen")
        return variance * (self.width ** 2)

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

    def __hash__(self):
        return super().__hash__()

    def normalize_point(self, point):
        """
        Get a prediciton sample value on the normalized scale from a true-scale value

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
        return (point * self.width) + self.low

    def destructure(self):
        return ((LogScale,), (self.low, self.high, self.log_base))

    @classmethod
    def structure(cls, params):
        low, high, log_base = params
        return cls(low, high, log_base)


@dataclass
class TimeScale(Scale):
    def __init__(self, low, high):

        self.low = low
        self.high = high
        self.width = self.high - self.low

    def __repr__(self):
        return f"TimeScale(low={self.timestamp_to_str(self.low)}, high={self.timestamp_to_str(self.high)}, width={timedelta(seconds=self.width)})"

    def __hash__(self):
        return super().__hash__()

    def destructure(self):
        return (
            (TimeScale,),
            (self.low, self.high,),
        )

    def timestamp_to_str(self, timestamp: float) -> str:
        return time.strftime(
            "%Y-%m-%d", time.localtime(timestamp)
        )  # expand this for datetimes if desirable


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
