from datetime import datetime

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
        raise KeyError(f"value for county and state: {county} not in the projections")
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
