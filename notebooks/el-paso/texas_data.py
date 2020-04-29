import pandas as pd
from datetime import date


def get_el_paso_data():
    """
    Get El Paso COVID data from the Texas government's data at https://dshs.texas.gov/coronavirus/TexasCOVID19DailyCountyCaseCountData.xlsx
    """
    texas_cases = pd.read_excel(
        "https://dshs.texas.gov/coronavirus/TexasCOVID19DailyCountyCaseCountData.xlsx"
    )
    texas_cases.columns = texas_cases.iloc[1]

    el_paso_cases = (
        texas_cases.loc[texas_cases["County Name"] == "El Paso"]
        .drop(columns=["County Name", "Population"])
        .transpose()
    )

    el_paso_cases.columns = ["Cases so far"]

    def get_date(column_name):
        date_str = column_name.split("\n")[1]
        month_str, day_str = date_str.split("-")
        return date(2020, int(month_str), int(day_str))

    el_paso_cases.index = [get_date(id) for id in el_paso_cases.index]

    el_paso_cases["New cases"] = el_paso_cases["Cases so far"].diff()

    return el_paso_cases
