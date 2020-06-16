from datetime import date, timedelta
from typing import Callable

import pandas as pd

import sklearn


def get_hospital_stay_days():
    # from https://penn-chime.phl.io/
    hospital_stay_days_point_estimate = 7

    # get_hospital_confirmed_from_daily_infected_model causes random choices
    # (because hospital_stay_days is random)
    # but we’re calling it outside of the model functions so it’s only called once.
    # this will give weird/wrong results.
    # moving hospital_confirmed_from_daily_infected_model =
    # get_hospital_confirmed_from_daily_infected_model(daily_infections)
    # inside hospital_confirmed_for_date will fix this
    # but will make the model pretty slow since each call to the model reruns regression
    # hospital_stay_days_fuzzed = round(
    #     float(
    #         ergo.normal_from_interval(
    #             hospital_stay_days_point_estimate * 0.5,
    #             hospital_stay_days_point_estimate * 1.5,
    #         )
    #     )
    # )

    # return max(1, hospital_stay_days_fuzzed)

    return hospital_stay_days_point_estimate


def get_daily_hospital_confirmed(
    hospital_data: pd.DataFrame, daily_infections_fn: Callable[[date], int]
):
    """
    Use a linear regression to predict
    the number of patients with COVID currently in the hospital
    from the total number of new confirmed cases over the past several days

    :param data: dataframe with index of dates,
    columns of "In hospital confirmed"
    :return: A function to predict
    the number of confirmed cases of COVID in the hospital on a date,
    given the total number of confirmed cases for each date
    """

    hospital_stay_days = get_hospital_stay_days()

    has_hospital_confirmed = hospital_data[hospital_data["In hospital confirmed"].notna()]  # type: ignore

    data_dates = has_hospital_confirmed.index

    hospital_confirmed = has_hospital_confirmed["In hospital confirmed"]

    def get_recent_cases_data(date):
        """
        How many new confirmed cases were there over the past hospital_stay_days days?
        """
        return sum(
            [
                daily_infections_fn(date - timedelta(n))
                for n in range(0, hospital_stay_days)
            ]
        )

    recent_cases = [[get_recent_cases_data(date)] for date in data_dates]

    reg = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(
        recent_cases, hospital_confirmed
    )
    # TODO: consider adding uncertainty to the fit here

    # now that we've related current hospitalized cases and recent confirmed cases,
    # return a function that allows us to predict hospitalized cases given estimates
    # of future confirmed cases
    def get_hospital_confirmed_from_daily_cases(date: date):
        recent_cases = get_recent_cases_data(date)
        return round(reg.predict([[recent_cases]])[0])

    return get_hospital_confirmed_from_daily_cases
