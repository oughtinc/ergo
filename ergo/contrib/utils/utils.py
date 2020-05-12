from datetime import timedelta
from typing import Union

import numpy as np
import pandas as pd

import ergo

ArrayLikeType = Union[pd.DataFrame, pd.Series, np.ndarray]


## BEGIN UTILS ##
# TODO consider if these belong in question classes or perhaps in ergo proper


def sample_from_ensemble(
    models, params, weights=None, fallback=False, noise_sd=False, default=None
):
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
        elif noise_sd:
            result = ergo.normal(mean=result, stdev=result * noise_sd)
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


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def rejection_sample(fn, condition):
    """
    Sample from fn until we get a value that satisfies
    condition, then return it.
    """
    while True:
        value = fn()
        if condition(value):
            return value
