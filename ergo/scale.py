from dataclasses import dataclass, field
from typing import TypeVar

import jax.numpy as np


@dataclass
class Scale:
    scale_min: float
    scale_max: float
    scale_range: float = field(init=False)

    def __post_init__(self):
        self.scale_range = self.scale_max - self.scale_min

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Scale):
            return self.__key() == other.__key()
        return NotImplemented

    def __key(self):
        cls, params = self.destructure()
        return (cls, params)

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

    def denormalize_points(self, points):
        return [self.denormalize_point(point) for point in points]

    def normalize_points(self, points):
        return [self.normalize_point(point) for point in points]

    def normalize_variance(self, variance):
        if variance is None:
            raise Exception("Point was None This shouldn't happen")
        return variance / (self.scale_range ** 2)

    def denormalize_variance(self, variance):
        if variance is None:
            raise Exception("Point was None This shouldn't happen")
        return variance * (self.scale_range ** 2)

    def destructure(self):
        return (Scale, (self.scale_min, self.scale_max))

    def export(self):
        cls, params = self.destructure()
        return (cls.__name__, params)


ScaleClass = TypeVar("ScaleClass", bound=Scale)


@dataclass
class LogScale(Scale):
    log_base: float

    def __post_init__(self):
        self.scale_range = self.scale_max - self.scale_min

    def __hash__(self):
        return super.__hash__(self)

    # TODO do we still need this default?
    def normalize_point(self, point):
        """
        Get a prediciton sample value on the normalized scale from a true-scale value

        :param true_value: a sample value on the true scale
        :return: a sample value on the normalized scale
        """
        if point is None:
            raise Exception("Point was None This shouldn't happen")
        shifted = point - self.scale_min
        numerator = shifted * (self.log_base - 1)
        scaled = numerator / self.scale_range
        timber = 1 + scaled
        floored_timber = np.amax([timber, 1e-9])

        return np.log(floored_timber) / np.log(self.log_base)

    # TODO do we still need this default?
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
        scaled = self.scale_range * deriv_term
        return self.scale_min + scaled
        return (point * self.scale_range) + self.scale_min

    def destructure(self):
        return (LogScale, (self.scale_min, self.scale_max, self.log_base))


class DateScale(Scale):
    def __post_init__(self):
        self.scale_range = self.scale_max - self.scale_min

    def __hash__(self):
        return super.__hash__(self)

    # TODO do we still need this default?
    def normalize_point(self, point):
        """
        Get a prediciton sample value on the normalized scale from a true-scale value

        :param true_value: a sample value on the true scale
        :return: a sample value on the normalized scale
        """
        if point is None:
            raise Exception("Point was None This shouldn't happen")
        shifted = point - self.scale_min
        numerator = shifted * (self.log_base - 1)
        scaled = numerator / self.scale_range
        timber = 1 + scaled
        floored_timber = np.amax([timber, 1e-9])

        return np.log(floored_timber) / np.log(self.log_base)

    # TODO do we still need this default?
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
        scaled = self.scale_range * deriv_term
        return self.scale_min + scaled
        return (point * self.scale_range) + self.scale_min

    def destructure(self):
        return (LogScale, (self.scale_min, self.scale_max, self.log_base))


def scale_factory(class_name, params):
    if type(class_name) == str:
        if class_name == "Scale":
            return Scale(*params)
        elif class_name == "LogScale":
            return LogScale(*params)
    # elif isinstance(class_name, type) and issubclass(class_name, Scale):
    #     return class_name(*params)
    print("cannot reconstruct Scale")
