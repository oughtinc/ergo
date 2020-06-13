from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Tuple, TypeVar, Union

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

    def destructure(self):
        return (Scale, (self.low, self.high))

    @classmethod
    def structure(cls, params):
        low, high = params
        return cls(low, high)

    def export(self):
        cls, params = self.destructure()
        return (cls.__name__, params)


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
        scaled = self.width * deriv_term
        return self.low + scaled
        return (point * self.width) + self.low

    def destructure(self):
        return (LogScale, (self.low, self.high, self.log_base))

    @classmethod
    def structure(cls, params):
        low, high, log_base = params
        return cls(low, high, log_base)


@dataclass
class TimeScale(Scale):
    time_unit: str  # function that returns timedelta in desired scale units e.g. seconds, days, months, years
    time_units: Tuple[str, str, str, str, str, str] = (
        "years",
        "months",
        "days",
        "hours",
        "seconds",
        "microseconds",
    )

    def __init__(self, low, high, time_unit):
        if time_unit not in self.time_units:
            raise Exception(
                "time_unit needs to be one of 'years','months','days','hours','seconds', 'microseconds'"
            )
        self.time_unit = time_unit

        self.low = self.string_to_datetime(low) if isinstance(low, str) else low
        self.high = self.string_to_datetime(high) if isinstance(high, str) else high
        if not isinstance(self.low, (date, datetime)) and not isinstance(
            self.high, (date, datetime)
        ):
            raise Exception(
                "Low value {low} and High value {high} could not be parsed into Datetime "
            )

        # self.low_string = self.low.isoformat()
        # self.high_string = self.high.isoformat()

        self.width = getattr(self.high - self.low, self.time_unit)

    def __hash__(self):
        return super.__hash__(self)

    def normalize_point(self, point: Union[str, date, datetime]) -> float:
        """
        Get a prediciton point on the normalized scale from a true-scale value

        :param point: a point on the true scale
        :return: a sample value on the normalized scale
        """
        if isinstance(point, str):
            point = self.string_to_datetime(point)

        return float(getattr(point - self.low, self.time_unit) / self.width)  # type: ignore

    def denormalize_point(
        self, point: float, as_string: bool = True
    ) -> Union[str, date, datetime]:
        """
        Get a value on the true scale from a normalized-scale value

        :param point: a point on the normalized scale
        :return: a point on the true scale
        """
        denormed_point = self.low + timedelta(  # type: ignore
            **{self.time_unit: round(self.width * point)}
        )
        return denormed_point.isoformat() if as_string else denormed_point

    def string_to_datetime(self, date_string: str) -> Union[date, datetime]:
        try:
            return parse(date_string)
        except ValueError as e:
            print(str(e))
            raise TypeError(
                "Date format must be in ISO format: e.g. '2011-11-04' or '2011-11-04 00:05:23.283+00:00'"
            )

    @classmethod
    def structure(cls, params):
        low, high, time_unit = params
        return cls(
            datetime.fromtimestamp(low),
            datetime.fromtimestamp(high),
            cls.time_units[time_unit],
        )

    def destructure(self):
        return (
            TimeScale,
            (
                datetime(*self.low.timetuple()[:6]).timestamp(),
                datetime(*self.high.timetuple()[:6]).timestamp(),
                self.time_units.index(self.time_unit),
            ),
        )


def scale_factory(class_name, params):
    if type(class_name) == str:
        if class_name == "Scale":
            return Scale.structure(params)
        elif class_name == "LogScale":
            return LogScale.structure(params)
        elif class_name == "TimeScale":
            return TimeScale.structure(params)
    raise TypeError("cannot reconstruct Scale")
