from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
import time
from typing import TypeVar, Union

from dateutil.parser import parse
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

    def denormalize_points(self, points, **kwargs):
        return [self.denormalize_point(point, **kwargs) for point in points]

    def normalize_points(self, points, **kwargs):
        return [self.normalize_point(point, **kwargs) for point in points]

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

    @classmethod
    def structure(cls, params):
        low, high = params
        return cls(low, high)

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

        shifted = point - self.low
        numerator = shifted * (self.log_base - 1)
        scaled = numerator / self.width
        timber = 1 + scaled
        floored_timber = np.maximum(timber, 1e-9)

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
    def __init__(self, low, high, direct_init=False):
        if direct_init:
            self.low = low
            self.high = high
            self.width = self.high - self.low
        else:
            if isinstance(low, (float, int)):
                self.low = low
            elif isinstance(low, str):
                self.low = self.str_to_timestamp(low)
            elif isinstance(low, (date, datetime)):
                self.low = self.datetime_to_timestamp(low)
            if isinstance(high, (float, int)):
                self.high = high
            elif isinstance(high, str):
                self.high = self.str_to_timestamp(high)
            elif isinstance(high, (date, datetime)):
                self.high = self.datetime_to_timestamp(high)
            self.width = self.high - self.low

            assert isinstance(self.low, float), f"low was {self.low}"
            assert isinstance(self.high, float), f"high was {self.high}"
            assert isinstance(self.width, float), f"widht was {self.width}"

    def __repr__(self):
        return f"TimeScale(low={self.timestamp_to_str(self.low)}, high={self.timestamp_to_str(self.high)}, width={timedelta(seconds=self.width)})"

    def __hash__(self):
        return super.__hash__(self)

    def normalize_point(self, point) -> float:
        """
        Get a prediciton point on the normalized scale from a true-scale value

        :param point: a point on the true scale
        :return: a sample value on the normalized scale
        """
        if isinstance(point, str):
            point = self.str_to_timestamp(point)
        if isinstance(point, (date, datetime)):
            point = self.datetime_to_timestamp(point)

        assert isinstance(point, float)
        return (point - self.low) / self.width  # type: ignore

    def denormalize_point(
        self, point: float, as_string: bool = True
    ) -> Union[str, float]:
        """
        Get a value on the true scale from a normalized-scale value

        :param point: a point on the normalized scale
        :return: a point on the true scale
        """
        denormed_point = self.low + self.width * point  # type: ignore
        assert isinstance(denormed_point, float)
        return self.timestamp_to_str(denormed_point) if as_string else denormed_point

    def str_to_datetime(self, date_string: str) -> datetime:
        try:
            return parse(date_string)
        except ValueError as e:
            print(str(e))
            raise TypeError(
                "Datetimes in string format must be in ISO format: e.g. '2011-11-04' or '2011-11-04 00:05:23.283+00:00'"
            )

    def str_to_timestamp(self, date_string: str) -> float:
        return self.str_to_datetime(date_string).timestamp()

    def datetime_to_timestamp(self, xdatetime: Union[date, datetime]) -> float:
        if isinstance(xdatetime, datetime):
            return xdatetime.timestamp()
        else:
            return datetime(*xdatetime.timetuple()[:6]).timestamp()

    def timestamp_to_str(self, timestamp: float) -> str:
        return time.strftime(
            "%Y-%m-%d", time.localtime(timestamp)
        )  # expand this for datetimes if desirable

    def destructure(self):
        return (
            TimeScale,
            (self.low, self.high,),
        )

    @classmethod
    def structure(cls, params):
        low, high = params
        low = low + 0.0
        high = high + 0.0
        return cls(low, high, direct_init=True)



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


def scale_factory_II(class_name, params):
    if type(class_name) == str:
        if class_name == "Scale":
            return Scale.structure(params)
        elif class_name == "LogScale":
            return LogScale.structure(params)
        elif class_name == "TimeScale":
            return TimeScale.structure(params)
    raise TypeError("cannot reconstruct Scale")

