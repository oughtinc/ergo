from datetime import date, datetime, timedelta

import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_histogram,
    ggplot,
    guides,
    scale_fill_brewer,
    scale_x_datetime,
    theme,
)

from ergo.theme import ergo_theme

from .linear import LinearQuestion


class LinearDateQuestion(LinearQuestion):
    # TODO: add log functionality (if some psychopath makes a log scaled date question)

    def _scale_x(self, xmin: float = None, xmax: float = None):
        return scale_x_datetime(limits=(xmin, xmax))

    @property
    def question_range(self):
        """
        Question range from the Metaculus data plus the question's data range
        """
        qr = {
            "min": 0,
            "max": 1,
            "date_min": datetime.strptime(
                self.possibilities["scale"]["min"], "%Y-%m-%d"
            ).date(),
            "date_max": datetime.strptime(
                self.possibilities["scale"]["max"], "%Y-%m-%d"
            ).date(),
        }
        qr["date_range"] = (qr["date_max"] - qr["date_min"]).days
        return qr

    # TODO Make less fancy. Would be better to only accept datetimes
    def normalize_samples(self, samples):
        """
        Normalize samples from dates to the normalized scale used by the Metaculus API

        :param samples: dates from the predicted distribution answering the question
        :return: normalized samples
        """
        if isinstance(samples[0], date):
            if type(samples) != pd.Series:
                try:
                    samples = pd.Series(samples)
                except ValueError:
                    raise ValueError("Could not process samples vector")
            return self.normalize_dates(samples)
        else:
            return super().normalize_samples(samples)

    def normalize_dates(self, dates: pd.Series):
        """
        Map dates to the normalized scale used by the Metaculus API

        :param dates: a pandas series of dates
        :return: normalized samples
        """

        return (dates - self.question_range["date_min"]).dt.days / self.question_range[
            "date_range"
        ]

    def denormalize_samples(self, samples):
        """
        Map normalized samples to dates using the date range from the question

        :param samples: normalized samples
        :return: dates
        """

        def denorm(sample):
            return self.question_range["date_min"] + timedelta(
                days=round(self.question_range["date_range"] * sample)
            )

        if type(samples) == float:
            return denorm(samples)
        else:
            samples = pd.Series(samples)
            return samples.apply(denorm)

    # TODO enforce return type date/datetime
    def sample_community(self):
        """
        Sample an approximation of the entire current community prediction,
        on the true scale of the question.

        :return: One sample on the true scale
        """
        normalized_sample = self.sample_normalized_community()
        return self.denormalize_samples(normalized_sample)

    def comparison_plot(  # type: ignore
        self, df: pd.DataFrame, xmin=None, xmax=None, bins: int = 50, **kwargs
    ):

        return (
            ggplot(df, aes(df.columns[1], fill=df.columns[0]))
            + scale_fill_brewer(type="qual", palette="Pastel1")
            + geom_histogram(position="identity", alpha=0.9, bins=bins)
            + self._scale_x(xmin, xmax)
            + facet_wrap(df.columns[0], ncol=1)
            + guides(fill=False)
            + ergo_theme
            + theme(axis_text_x=element_text(rotation=45, hjust=1))
        )

    def density_plot(  # type: ignore
        self,
        df: pd.DataFrame,
        xmin=None,
        xmax=None,
        fill: str = "#fbb4ae",
        bins: int = 50,
        **kwargs,
    ):

        return (
            ggplot(df, aes(df.columns[0]))
            + geom_histogram(fill=fill, bins=bins)
            + self._scale_x(xmin, xmax)
            + ergo_theme
            + theme(axis_text_x=element_text(rotation=45, hjust=1))
        )
