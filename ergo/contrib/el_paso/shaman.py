import pandas as pd
import numpy as np
from datetime import date
from ergo.contrib.utils import daterange
from typing import Tuple
import ergo


def extract_projections_for_param(
    county: str, param: str, column_prefix: str, raw_us_projections
):
    raw_county_projections = raw_us_projections[raw_us_projections.county == county]
    metadata = raw_county_projections[["county", "fips", "Date"]]
    metadata["var"] = param
    percentiles = ["2.5", "25", "50", "75", "97.5"]
    projection_column_names = [
        f"{column_prefix}_{percentile}" for percentile in percentiles
    ]
    projections = raw_county_projections[projection_column_names]
    projections.columns = percentiles
    return pd.concat([metadata, projections], axis=1)


def load_cu_projections(county: str):
    """
    The COVID model from the Shaman lab at Columbia projects
    "daily new confirmed case, daily new infection (both reported and unreported),
    cumulative demand of hospital beds, ICU and ventilators
    as well as daily mortality (2.5, 25, 50, 75 and 97.5 percentiles)"
    (https://github.com/shaman-lab/COVID-19Projection)

    Load this data
    """
    # pd has this option but mypy doesn't know about it
    pd.options.mode.chained_assignment = None  # type: ignore
    scenarios = ["nointerv", "60contact", "70contact", "80contact"]
    cu_model_data = {}
    for scenario in scenarios:
        # pd.read_csv has a parse_dates option but mypy doesn't know about it
        raw_cases_df = pd.read_csv(  # type: ignore
            f"https://raw.githubusercontent.com/shaman-lab/COVID-19Projection/master/Projection_April26/Projection_{scenario}.csv",
            parse_dates=["Date"],
        )
        cases_projections_df = extract_projections_for_param(
            county, "cases", "report", raw_cases_df
        )

        raw_covid_effects_df = pd.read_csv(  # type: ignore
            f"https://raw.githubusercontent.com/shaman-lab/COVID-19Projection/master/Projection_April26/bed_{scenario}.csv",
            parse_dates=["Date"],
        )

        hospital_projections_df = extract_projections_for_param(
            county, "hosp", "hosp_need", raw_covid_effects_df
        )
        icu_projections_df = extract_projections_for_param(
            county, "ICU", "ICU_need", raw_covid_effects_df
        )
        vent_projections_df = extract_projections_for_param(
            county, "vent", "vent_need", raw_covid_effects_df
        )
        deaths_projections_df = extract_projections_for_param(
            county, "deaths", "death", raw_covid_effects_df
        )

        all_projections_df = pd.concat(
            [
                cases_projections_df,
                hospital_projections_df,
                icu_projections_df,
                vent_projections_df,
                deaths_projections_df,
            ]
        )
        all_projections_df["Date"] = all_projections_df["Date"].apply(
            lambda x: x.date()
        )
        cu_model_data[scenario] = all_projections_df
    return cu_model_data


# The below were added for the workflow/tutorial nb
# they're not yet used in the main El Paso notebook
@ergo.mem
def cu_model_scenario(scenarios: Tuple[str]):
    """Which of the model scenarios are we in?"""
    return ergo.random_choice(scenarios)


@ergo.mem
def cu_model_quantile():
    """Where in the distribution of model outputs are we for this model run?
    Want to be consistent across time, so we sample it once per model run"""
    return ergo.uniform()


def cu_projection(param: str, date: date, cu_projections) -> int:
    """
    Get the Columbia model's prediction
    of the param for the date
    """
    scenario = cu_model_scenario(tuple([s for s in cu_projections.keys()]))
    quantile = cu_model_quantile()

    # Extract quantiles of the model distribution
    xs = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
    scenario_df = cu_projections[scenario]
    param_df = scenario_df[scenario_df["var"] == param]
    date_df = param_df[param_df["Date"] == date]
    if date_df.empty:
        raise KeyError(f"No Columbia project for param: {param}, date: {date}")

    ys = np.array(date_df[["2.5", "25", "50", "75", "97.5"]].iloc[0])

    # Linearly interpolate
    # mypy doesn't know that there's an np.interp
    return int(round(np.interp(quantile, xs, ys)))  # type: ignore


def cu_projections_for_dates(
    param: str, start_date: date, end_date: date, cu_projections
):
    """
    Get Columbia model projections over a range of dates
    """
    date_range = daterange(start_date, end_date)
    return [cu_projection(param, date, cu_projections) for date in date_range]
