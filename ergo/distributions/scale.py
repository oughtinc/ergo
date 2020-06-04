from dataclasses import dataclass, field
import math
from typing import TypeVar

import jax.numpy as np


@dataclass
class Scale:
    scale_min: float
    scale_max: float
    scale_range: float = field(init=False)

    def __post_init__(self):
        self.scale_range = self.scale_max - self.scale_min
        self.bins = lambda size: np.linspace(self.scale_min, self.scale_max, size)

    def normalize_point(self, point, default=None):
        return (
            (point - self.scale_min) / self.scale_range
            if point is not None
            else default
        )

    def denormalize_point(self, point, default=None):
        return (
            (point * self.scale_range) + self.scale_min
            if point is not None
            else default
        )

    def destructure(self):
        return (Scale, (self.scale_min, self.scale_max))

    def export(self):
        cls, params = self.destructure()
        return (cls.__name__, params)


ScaleClass = TypeVar("ScaleClass", bound=Scale)


@dataclass
class LogScale(Scale):
    deriv_ratio: float
    display_base: float = 10

    def __post_init__(self):
        self.scale_range = self.scale_max - self.scale_min
        self.bins = lambda size: np.logspace(
            self.scale_min, self.scale_max, size, base=self.display_base
        )

    def normalize_point(self, point, default=None):
        """
        Get a prediciton sample value on the normalized scale from a true-scale value

        :param true_value: a sample value on the true scale
        :return: a sample value on the normalized scale
        """
        shifted = point - self.scale_min
        numerator = shifted * (self.deriv_ratio - 1)
        scaled = numerator / self.scale_range
        timber = 1 + scaled
        floored_timber = max(timber, 1e-9)
        return (
            math.log(floored_timber, self.deriv_ratio) if point is not None else default
        )

    def denormalize_point(self, point, default=None):
        """
        Get a value on the true scale from a normalized-scale value

        :param normalized_value: [description]
        :type normalized_value: [type]
        :return: [description]
        :rtype: [type]
        """
        deriv_term = (self.deriv_ratio ** point - 1) / (self.deriv_ratio - 1)
        scaled = self.scale_range * deriv_term
        return self.scale_min + scaled
        return (
            (point * self.scale_range) + self.scale_min
            if point is not None
            else default
        )

    def destructure(self):
        return (LogScale, (self.scale_min, self.scale_max, self.deriv_ratio))


def ScaleFactory(cls, params):
    if type(cls) == str:
        if cls == "Scale":
            return Scale(*params)
        elif cls == "LogScale":
            return LogScale(*params)
    elif isinstance(cls, type) and issubclass(cls, Scale):
        return cls(*params)
    print("cannot reconstruct Scale")
