from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import TypeVar, Union

import jax.numpy as np

from ergo.utils import trapz


@dataclass
class Scale:
    low: Union[date, datetime, float]
    high: Union[date, datetime, float]
    width: float = field(init=False)

    def __post_init__(self):
        self.width = self.high - self.low
        self.__norm_term = self.width

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Scale):
            return self.__key() == other.__key()
        return NotImplemented

    def __key(self):
        cls, params = self.destructure()
        return (cls, params)

    @property
    def norm_term(self):
        return self.__norm_term

    @norm_term.setter
    def norm_term(self, pairs):
        pass

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

    # TODO I'm not sure if we will need this anywhere
    def normalize_density(self, density):
        return density * self.norm_term

    def denormalize_density(self, density):
        return density / self.norm_term

    # TODO I think we can simply do this in the function inits, but perhaps having logic here is more consistent?
    def normalize_densities(self, densities):
        return densities * self.norm_term

    # TODO should this call normalized_density? It is probably faster this way...
    def denormalize_densities(self, densities):
        return densities / self.norm_term

    def copy(self):
        return self.structure(self.destructure())

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

    def __hash__(self):
        return super.__hash__(self)

    @property
    def norm_term(self):
        return self.__norm_term

    @norm_term.setter
    def norm_term(self, params):
        # The params are either normalized xs and denormalized densities or
        # denormalized xs and normalized densities.
        xs, densities, densities_normed = params
        if densities_normed:
            self.__norm_term = trapz(densities, x=xs)
        else:
            self.__norm_term = 1 / trapz(densities, x=xs)

    # TODO I think we can retire this:

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
