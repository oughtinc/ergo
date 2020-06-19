from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import TypeVar, Union

import jax.numpy as np


@dataclass
class Scale:
    low: Union[date, datetime, float]
    high: Union[date, datetime, float]
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

    @classmethod
    def structure(cls, params):
        classes, numeric = params
        return classes[0](*numeric)

    def destructure(self):
        return ((Scale,), (self.low, self.high))

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
        self.linear_scale = Scale(np.log(self.low), np.log(self.high))

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
        floored_point = np.maximum(point, 1e-9)
        return self.linear_scale.normalize_point(np.log(floored_point))

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
        return np.exp(self.linear_scale.denormalize_point(point))

    def destructure(self):
        return ((LogScale,), (self.low, self.high, self.log_base))


@dataclass
class TimeScale(Scale):
    time_unit: str  # function that returns timedelta in desired scale units e.g. seconds, days, months, years

    def __post_init__(self):
        self.width = getattr(self.high - self.low, self.time_unit)

    def __hash__(self):
        return super.__hash__(self)

    def normalize_point(self, point: Union[date, datetime]) -> float:
        """
        Get a prediciton point on the normalized scale from a true-scale value

        :param point: a point on the true scale
        :return: a sample value on the normalized scale
        """
        return float(getattr(point - self.low, self.time_unit) / self.width)  # type: ignore

    def denormalize_point(self, point: float) -> Union[date, datetime]:
        """
        Get a value on the true scale from a normalized-scale value

        :param point: a point on the normalized scale
        :return: a point on the true scale
        """
        return self.low + timedelta(  # type: ignore
            **{self.time_unit: round(self.width * point)}
        )

    def destructure(self):
        return ((TimeScale,), (self.low, self.high, self.time_unit))


def scale_factory(scale_dict):
    scale_class = scale_dict["class"]
    low = scale_dict["low"]
    high = scale_dict["high"]

    if scale_class == "Scale":
        return Scale(low, high)
    if scale_class == "LogScale":
        return LogScale(low, high, scale_dict["log_base"])
    raise NotImplementedError(
        f"reconstructing scales of class {scale_class} is not implemented."
    )
