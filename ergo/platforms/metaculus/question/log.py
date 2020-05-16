import math

from plotnine import scale_x_log10

from .continuous import ContinuousQuestion


class LogQuestion(ContinuousQuestion):
    @property
    def deriv_ratio(self) -> float:
        return self.possibilities["scale"]["deriv_ratio"]

    def _scale_x(self, xmin: float = None, xmax: float = None):
        return scale_x_log10(limits=(xmin, xmax))

    def normalized_from_true_value(self, true_value) -> float:
        """
        Get a prediciton sample value on the normalized scale from a true-scale value

        :param true_value: a sample value on the true scale
        :return: a sample value on the normalized scale
        """
        shifted = true_value - self.question_range["min"]
        numerator = shifted * (self.deriv_ratio - 1)
        scaled = numerator / self.question_range_width
        timber = 1 + scaled
        floored_timber = max(timber, 1e-9)
        return math.log(floored_timber, self.deriv_ratio)

    def true_from_normalized_value(self, normalized_value):
        """
        Get a prediciton sample value on the true scale from a normalized-scale value

        :param normalized_value: [description]
        :type normalized_value: [type]
        :return: [description]
        :rtype: [type]
        """
        deriv_term = (self.deriv_ratio ** normalized_value - 1) / (self.deriv_ratio - 1)
        scaled = self.question_range_width * deriv_term
        return self.question_range["min"] + scaled

    def normalize_samples(self, samples):
        """
        Map samples from the true scale to the normalized scale

        :param samples: Samples on the true scale
        :return: Samples on the normalized scale
        """
        return [self.normalized_from_true_value(sample) for sample in samples]

    def denormalize_samples(self, samples):
        """
        Map samples from the normalized scale to the true scale

        :param samples: Samples on the normalized scale
        :return: Samples on the true scale
        """
        return [self.true_from_normalized_value(sample) for sample in samples]
