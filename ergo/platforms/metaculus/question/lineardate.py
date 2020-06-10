import datetime
from typing import Any, Dict

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

from ergo.scale import TimeScale
from ergo.theme import ergo_theme

from .linear import LinearQuestion


class LinearDateQuestion(LinearQuestion):
    scale: TimeScale

    def __init__(
        self, id: int, metaculus: Any, data: Dict, name=None,
    ):
        super().__init__(id, metaculus, data, name)

        self.scale = TimeScale(
            datetime.datetime.strptime(
                self.possibilities["scale"]["min"], "%Y-%m-%d"
            ).date(),
            datetime.datetime.strptime(
                self.possibilities["scale"]["max"], "%Y-%m-%d"
            ).date(),
            "days",
        )

    def _scale_x(self, xmin: float = None, xmax: float = None):
        return scale_x_datetime(limits=(xmin, xmax))

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
