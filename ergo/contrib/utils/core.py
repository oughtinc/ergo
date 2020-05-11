from datetime import date, timedelta
import functools
from typing import Union

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

# ArrayLikes = [pd.DataFrame, pd.Series, np.ndarray]
ArrayLikeType = Union[pd.DataFrame, pd.Series, np.ndarray]


# TODO consider turning this into a Class or factoring into Ergo proper


def rejection_sample(fn, condition):
    """
    Sample from fn until we get a value that satisfies
    condition, then return it.
    """
    while True:
        value = fn()
        if condition(value):
            return value


# Associate models with questions

# We'll add a sampler here for each question we predict on.
# Each sampler is a function that returns a single sample
# from our model predicting on that question.
samplers = {}


# TODO probably curry this with the notbooks metaculus instance so we don't need to pass
# it in on every question
def question(
    metaculus,
    question_id,
    community_weight=0,
    community_fn=None,
    start_date=date.today(),
):
    q = metaculus.get_question(question_id)

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
            if isinstance(value, date):
                # FIXME: Ergo needs to handle dates
                ergo.tag(int((value - start_date).days), tag)
            else:
                ergo.tag(value, tag)
            return value

        sampler.question = q
        samplers[q.id] = sampler
        return sampler

    return decorator


def summarize_question_samples(samples):
    sampler_tags = [sampler.__name__ for sampler in samplers.values()]
    tags_to_show = [tag for tag in sampler_tags if tag in samples.columns]
    samples_to_show = samples[tags_to_show]
    summary = samples_to_show.describe().transpose().round(2)
    display(summary)  # noqa: F821   #TODO see if we need this display command


def plot_question(sampler, num_samples=200, bw=None, start_date=date.today()):
    samples = ergo.run(sampler, num_samples=num_samples)

    summarize_question_samples(samples)

    q = sampler.question

    q_samples = samples[sampler.__name__]

    if (
        q.id == 4128
    ):  # Date question: Need to convert back to date from days (https://github.com/oughtinc/ergo/issues/144)
        q_samples = np.array([start_date + timedelta(s) for s in q_samples])

    if bw is not None:
        q.show_prediction(
            samples=q_samples, show_community=True, percent_kept=0.9, bw=bw
        )
    else:
        q.show_prediction(samples=q_samples, show_community=True, percent_kept=0.9)


def sample_from_ensemble(models, params, weights=None, fallback=False, default=None):
    """Sample models in proportion to weights and execute with
    model_params. If fallback is true then call different model from
    ensemble if the selected model throws an error. If Default is not
    None then return default if all models fail

    """
    if len(models) > 1:
        model = ergo.random_choice(models, weights)
    else:
        model = models[0]
    try:
        result = model(**params)
        if np.isnan(result):
            raise KeyError
        return result
    except (KeyError, IndexError):
        if fallback and len(models) > 1:
            models_copy = models.copy()
            weights_copy = weights.copy()
            i = models.index(model)
            del models_copy[i]
            del weights_copy[i]
            return sample_from_ensemble(
                models_copy, params, weights_copy, fallback, default
            )
        return default


def get_central_quantiles(
    df: ArrayLikeType, percent_kept: float = 0.95, side_cut_from: str = "both",
):
    """
    Get the values that bound the central (percent_kept) of the sample distribution,
        i.e.,  cutting the tails from these values will give you the central.
        If passed a dataframe with multiple variables, the bounds that encompass
        all variables will be returned.

        :param df: pandas dataframe of one or more column of samples
        :param percent_kept: percentage of sample distrubtion to keep
        :param side_cut_from: which side to cut tails from,
            either 'both','lower', or 'upper'
        :return: lower and upper values of the central (percent_kept) of
            the sample distribution.
        """

    if side_cut_from not in ("both", "lower", "upper"):
        raise ValueError("side keyword must be either 'both','lower', or 'upper'")

    percent_cut = 1 - percent_kept
    if side_cut_from == "lower":
        _lb = percent_cut
        _ub = 1.0
    elif side_cut_from == "upper":
        _lb = 0.0
        _ub = 1 - percent_cut
    else:
        _lb = percent_cut / 2
        _ub = 1 - percent_cut / 2

    if isinstance(df, (pd.Series, np.ndarray)):
        _lq, _uq = df.quantile([_lb, _ub])  # type: ignore
        return (_lq, _uq)

    _lqs = []
    _uqs = []
    for col in df:
        _lq, _uq = df[col].quantile([_lb, _ub])
        _lqs.append(_lq)
        _uqs.append(_uq)
    return (min(_lqs), max(_uqs))


def plot_distribution(sampler, variable_name, num_samples=200, **kwargs):
    samples = ergo.run(sampler, num_samples=num_samples)
    show_distribution(
        samples=samples.iloc[:, 0],
        variable_name=variable_name,
        percent_kept=0.9,
        **kwargs,
    )


def show_distribution(
    samples,
    percent_kept: float = 0.95,
    side_cut_from: str = "both",
    num_samples: int = 1000,
    variable_name: str = "Variable",
    log=False,
    fill: str = "#fbb4ae",
    bw="normal_reference",
    bins: int = 50,
    **kwargs,
):
    """
    Plot samples of a distribution
    :param samples: 1-d array of samples from a distribution answering the prediction
     question (true scale).
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

    df = pd.DataFrame(data={"samples": samples})  # type:g ignore

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
    plot = plot + labs(
        x=variable_name, y="Density", title=f"Distribution of {variable_name}"
    )

    try:
        plot.draw()  # type: ignore
    except RuntimeError as err:
        print(err)
        print(
            "The plot was unable to automatically determine a bandwidth. You can manually specify one with the keyword 'bw', e.g., show_prediction(..., bw=.1)"
        )
