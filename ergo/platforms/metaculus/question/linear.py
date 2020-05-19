import jax.numpy as np

from ergo.distributions import Logistic, LogisticMixture

from .continuous import ContinuousQuestion


class LinearQuestion(ContinuousQuestion):
    """
    A continuous Metaculus question that's on a linear (as opposed to a log) scale"
    """

    def normalize_samples(self, samples):
        """
        Map samples from their true scale to the Metaculus normalized scale

        :param samples: samples from a distribution answering the prediction question
            (true scale)
        :return: samples on the normalized scale
        """
        return (samples - self.question_range["min"]) / (self.question_range_width)

    def denormalize_samples(self, samples):
        """
        Map samples from the Metaculus normalized scale to the true scale

        :param samples: samples on the normalized scale
        :return: samples from a distribution answering the prediction question
            (true scale)
        """

        # in case samples are in some other array-like format
        samples = np.array(samples)
        return self.question_range["min"] + (self.question_range_width) * samples

    # TODO: also return low and high on the true scale,
    # and use those somehow in logistic.py
    def get_true_scale_logistic(self, normalized_dist: Logistic) -> Logistic:
        """
        Convert a normalized logistic distribution to a logistic on
        the true scale of the question.

        :param normalized_dist: normalized logistic distribution
        :return: logistic distribution on the true scale of the question
        """
        true_loc = (
            normalized_dist.loc * self.question_range_width + self.question_range["min"]
        )

        true_scale = normalized_dist.scale * self.question_range_width
        return Logistic(true_loc, true_scale)

    def get_true_scale_mixture(
        self, normalized_dist: LogisticMixture
    ) -> LogisticMixture:
        """
        Convert a normalized logistic mixture distribution to a
        logistic on the true scale of the question.

        :param normalized_dist: normalized logistic mixture dist
        :return: same distribution rescaled to the true scale of the question
        """
        true_scale_logistics = [
            self.get_true_scale_logistic(c) for c in normalized_dist.components
        ]
        return LogisticMixture(true_scale_logistics, normalized_dist.probs)
