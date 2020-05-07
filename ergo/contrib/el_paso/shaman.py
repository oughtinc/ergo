import pandas as pd


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
