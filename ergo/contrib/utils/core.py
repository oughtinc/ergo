from datetime import date
import functools

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_density,
    geom_histogram,
    ggplot,
    labs,
    scale_x_continuous,
    scale_x_datetime,
    scale_x_log10,
    theme,
)

import ergo

from .utils import get_central_quantiles


def getNotebookQuestions(metaculus: ergo.MetaculusQuestion = None):
    if metaculus:
        return MetaculusNotebookQuestion(metaculus)
    else:
        return GenericNotebookQuestion()


class NotebookQuestion:
    """
    A question to be explored with ergo models with convenience functions for
    notebook-based development
    """

    def __init__(self):
        # Associate models with questions

        # We'll add a sampler here for each question we predict on.
        # Each sampler is a function that returns a single sample
        # from our model predicting on that question.

        self.samplers = {}

    def summarize_question_samples(self, samples):
        sampler_tags = [sampler.__name__ for sampler in self.samplers.values()]
        tags_to_show = [
            tag for tag in sampler_tags if tag in samples.columns
        ]  # when is this helpful?
        samples_to_show = samples[tags_to_show]
        summary = samples_to_show.describe().transpose().round(2)
        display(summary)  # noqa: F821


class GenericNotebookQuestion(NotebookQuestion):
    """
    A question without associations to a prediction market
    """

    def __init__(self):
        super().__init__()

    def question(self, name: str = None):
        def decorator(func):
            if name:
                tag = name
            else:
                tag = func.__name__

            @functools.wraps(func)
            @ergo.mem
            def sampler():
                value = func()
                # TODO we don't seem to need to go to numeric date encoding.
                #      See what changed.
                # if isinstanceo(value, date):
                #     # FIXME: Ergo needs to handle dates
                #     ergo.tag(int((value - start_date).days), tag)
                # else:
                ergo.tag(value, tag)
                return value

            self.samplers[tag] = sampler
            return sampler

        return decorator

    def plot_question(self, sampler, num_samples=200, start_date=None, **kwargs):
        variable_name = sampler.__name__
        samples = ergo.run(sampler, num_samples=num_samples)
        self.summarize_question_samples(samples)

        # TODO we don't seem to need to go to numeric date encoding.
        #      See what changed.
        # if start_date:
        #     # Date question: Need to convert back to date from days
        #       (https://github.com/oughtinc/ergo/issues/144)
        #     q_samples = np.array([start_date + timedelta(s) for s in q_samples])

        self.show_distribution(
            samples=samples[variable_name], variable_name=variable_name, **kwargs,
        )

    def show_distribution(
        self,
        samples,
        percent_kept: float = 0.9,
        side_cut_from: str = "both",
        num_samples: int = 1000,
        variable_name: str = "Variable",
        log=False,
        fill: str = "#b3cde3",
        bw="normal_reference",
        bins: int = 50,
        **kwargs,
    ):
        """
        Plot samples of a distribution
        :param samples: 1-d array of samples from a distribution answering the
            prediction question (true scale).
        :param percent_kept: percentage of sample distrubtion to keep
        :param side_cut_from: which side to cut tails from,
            either 'both','lower', or 'upper'
        :param num_samples: number of samples from the community
        :param **kwargs: additional plotting parameters
        """

        if isinstance(samples, list):
            samples = pd.Series(samples)
        if not isinstance(samples, (np.ndarray, pd.Series)):
            raise ValueError("Samples should be a list, numpy array or pandas series")

        df = pd.DataFrame(data={"samples": samples})  # type: ignore
        if isinstance(df.iloc[0, 0], date):
            plot = (
                ggplot(df, aes(df.columns[0]))
                + geom_histogram(fill=fill, bins=bins)
                + scale_x_datetime()
                + ergo.ergo_theme
                + theme(axis_text_x=element_text(rotation=45, hjust=1))
            )
        else:
            xmin, xmax = get_central_quantiles(
                df, percent_kept=percent_kept, side_cut_from=side_cut_from,
            )
            if log:
                scale_x = scale_x_log10
            else:
                scale_x = scale_x_continuous

            plot = (
                ggplot(df, aes(df.columns[0]))
                + geom_density(bw=bw, fill=fill, alpha=0.8)
                + scale_x(limits=(xmin, xmax))
                + ergo.ergo_theme
            )
        plot = plot + labs(x=variable_name, y="Density", title=f"Prediction")

        try:
            plot.draw()  # type: ignore
        except RuntimeError as err:
            print(err)
            print(
                "The plot was unable to automatically determine a bandwidth. You can manually specify one with the keyword 'bw', e.g., show_prediction(..., bw=.1)"
            )


class MetaculusNotebookQuestion(NotebookQuestion):
    """
    A question associated with the Metaculus prediction market
    """

    def __init__(self, metaculus: ergo.MetaculusQuestion):
        self.metaculus = metaculus
        super().__init__()

    def question(
        self,
        question_id,
        community_weight=0,
        community_fn=None,
        # start_date=date.today(),
    ):
        q = self.metaculus.get_question(question_id)

        def decorator(func):
            tag = func.__name__

            @functools.wraps(func)
            @ergo.mem
            def sampler():
                if ergo.flip(community_weight):
                    if community_fn:
                        value = community_fn()
                    else:
                        value = q.sample_community()
                else:
                    value = func()
                # if isinstance(value, date):
                #     # FIXME: Ergo needs to handle dates
                #     ergo.tag(int((value - start_date).days), tag)
                # else:
                ergo.tag(value, tag)
                return value

            sampler.question = q
            self.samplers[q.id] = sampler
            return sampler

        return decorator

    def plot_question(self, sampler, num_samples=200, start_date=None, **kwargs):
        samples = ergo.run(sampler, num_samples=num_samples)
        # self.summarize_question_samples(samples)

        q = sampler.question

        q_samples = samples[sampler.__name__]

        # if q.id == 4128 or start_date:
        #     # Date question: Need to convert back to date from days
        #            (https://github.com/oughtinc/ergo/issues/144)
        #     q_samples = np.array([start_date + timedelta(s) for s in q_samples])

        q.show_prediction(
            samples=q_samples, show_community=True, percent_kept=0.9, **kwargs
        )
