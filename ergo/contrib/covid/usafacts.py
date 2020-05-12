from datetime import datetime

import numpy as np
import pandas as pd


def load_county_data(county, state):
    us_cases = pd.read_csv(
        "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv"
    )

    if county in us_cases["County Name"].values and state in us_cases["State"].values:
        cases = us_cases[
            (us_cases["County Name"] == county) & (us_cases["State"] == state)
        ]
    else:
        raise KeyError(
            f"value for county: {county} and state: {state} were not in the data"
        )
    cases = pd.melt(
        cases,
        id_vars=["countyFIPS", "County Name", "State", "stateFIPS"],
        var_name="Date",
        value_name="cases",
    )
    cases["Date"] = cases["Date"].apply(
        lambda x: datetime.strptime(x, "%m/%d/%y").date()
    )
    return cases.set_index("Date")


def get_hospitalization_rate(state: str = None, date: str = None):
    state_data = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/05-10-2020.csv"
    )
    if state in state_data["Province_State"].values:
        hosp = state_data.loc[
            state_data["Province_State"] == "Washington", "Hospitalization_Rate"  # type: ignore
        ].values[0]
        if not np.isnan(hosp):
            return hosp
    else:
        return state_data["Hospitalization_Rate"].mean()
