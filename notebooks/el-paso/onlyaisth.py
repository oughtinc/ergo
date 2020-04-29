import pandas as pd


def get_onlyasith_results():
    """
    Get results from Metaculus user @onlyasith's model of cases in El Paso
    See this sheet for more information: https://docs.google.com/spreadsheets/d/1L6pzFAEJ6MfnUwt-ea6tetKyvdi0YubnK_70SGm436c/edit#gid=1807978187
    """
    projected_cases = pd.read_csv(
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSurcOWEsa7DBCRfONFA2Gxf802Rj1FebYSyVzvACysenRcD79Fs0ykXWJakIhGcW48_ymgw35TKga-/pub?gid=1213113172&single=true&output=csv",
        index_col="Date",
        parse_dates=True,
    )

    projected_cases = projected_cases.dropna()
    projected_cases["Cases so far"] = projected_cases["Cases so far"].apply(
        lambda str: int(str.replace(",", ""))
    )
    projected_cases["New cases"] = projected_cases["Cases so far"].diff()

    return projected_cases
