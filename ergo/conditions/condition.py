from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Tuple

import jax.numpy as np
import numpy as onp

import ergo.static as static


def static_value(v):
    if isinstance(v, np.DeviceArray) or isinstance(v, onp.ndarray):
        return tuple(v)
    else:
        return v


class Condition(ABC):
    weight: float = 1.0

    def __init__(self, weight=1.0):
        self.weight = weight

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Condition):
            return self.__key() == other.__key()
        return NotImplemented

    def __key(self):
        cls, params = self.destructure()
        return (cls, tuple(static_value(param) for param in params))

    def _describe_fit(self, dist) -> Dict[str, Any]:
        # convert to float for easy serialization
        return {"loss": self.loss(dist)}

    def normalize(self, scale_min: float, scale_max: float):
        """
        Assume that the condition's true range is [scale_min, scale_max].
        Return the normalized condition.

        :param scale_min: the true-scale minimum of the range
        :param scale_max: the true-scale maximum of the range
        :return: the condition normalized to [0,1]
        """
        return self

    def denormalize(self, scale_min: float, scale_max: float):
        """
        Assume that the condition has been normalized to be over [0,1].
        Return the condition on the true scale.

        :param scale_min: the true-scale minimum of the range
        :param scale_max: the true-scale maximum of the range
        :return: the condition on the true scale of [scale_min, scale_max]
        """
        return self

    def describe_fit(self, dist) -> Dict[str, float]:
        """
        Describe how well the distribution meets the condition

        :param dist: A probability distribution
        :return: A description of various aspects of how well
        the distribution meets the condition
        """
        result = static.describe_fit(*dist.destructure(), *self.destructure())
        return {k: float(v) for (k, v) in result.items()}

    @abstractmethod
    def loss(self, dist):
        """
        Loss function for this condition when fitting a distribution.

        Should have max loss = 1 without considering weight
        Should multiply loss * weight

        :param dist: A probability distribution
        """

    def shape_key(self):
        return (self.__class__.__name__,)

    @abstractmethod
    def destructure(self) -> Tuple["Condition", Sequence[Any]]:
        ...

    @classmethod
    def structure(cls, params) -> "Condition":
        return cls(*params)
