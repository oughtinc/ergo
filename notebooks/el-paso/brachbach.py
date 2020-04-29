from datetime import date, timedelta
from typing import Callable, List
import ergo
import pandas as pd
import sklearn


def get_hospital_stay_days():
    # from https://penn-chime.phl.io/
    hospital_stay_days_point_estimate = 7

    hospital_stay_days_fuzzed = round(
        float(
            ergo.normal_from_interval(
                hospital_stay_days_point_estimate * 0.5,
                hospital_stay_days_point_estimate * 1.5,
            )
        )
    )

    return max(1, hospital_stay_days_fuzzed)


def get_brachbach_daily_confirmed(data: pd.DataFrame):
    """
    Use a linear regression to predict
    the number of patients with COVID currently in the hospital
    from the total number of new confirmed cases over the past several days

    :param data: dataframe with index of dates,
    columns of "In hospital confirmed" and "New cases"
    :return: A function to predict
    the number of confirmed cases of COVID in the hospital on a date
    given the total number of confirmed cases for each date
    """

    hospital_stay_days = get_hospital_stay_days()

    has_hospital_confirmed = data[data["In hospital confirmed"].notna()]  # type: ignore

    data_dates = has_hospital_confirmed.index

    hospital_confirmed = has_hospital_confirmed["In hospital confirmed"]

    def get_recent_infected_data(date):
        """
        How many new confirmed cases were there over the past hospital_stay_days days?
        """
        return data[date - timedelta(hospital_stay_days) : date]["New cases"]

    recent_cases = [[get_recent_infected_data(date)] for date in data_dates]

    reg = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(
        recent_cases, hospital_confirmed
    )
    # TODO: consider adding uncertainty to the fit here

    # now that we've related current hospitalized cases and recent confirmed cases,
    # return a function that allows us to predict hospitalized cases given estimates
    # of future confirmed cases
    def hospital_confirmed_from_daily_cases(date: date, daily_infections_fn: Callable[[date], int]):
        recent_cases = sum(
            [
                daily_infections_fn(date - timedelta(n))
                for n in range(0, hospital_stay_days())
            ]
        )
        return round(reg.predict([[recent_cases]])[0])

def get_hospital_confirmed_predictor(hospital_confirmed_data: pd.DataFrame, daily_infections_fn: Callable[[date], int]):
    """
    Use data in the format from the Colab to get a function that predicts
    current confirmed cases in the hospital from
    recent confirmed cases
    """

    hospital_confirmed_data["New cases"] = hospital_confirmed_data.index daily_infections_fn()
    reg = get_brachbach_daily_confirmed(hospital_confirmed_data)
    


   

    
    

    return hospital_confirmed_from_daily_infected_model
