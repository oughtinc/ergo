import pandas as pd


def get_krismoore_data():
    """
    Get data from Metaculus user @oKrisMoore's compilation of El Paso COVID data
    See this sheet for more information:
    https://docs.google.com/spreadsheets/d/1eGF9xYmDmvAkr-dCmd-N4efHzPyYEfVl0YmL9zBvH9Q/edit#gid=1694267458
    """
    compiled_data = pd.read_csv(
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQEZk_8wZMF5MEm_f66wpev4nkWP7edQ8l6SwcbUd68zFZw6EVizh-jplw2_9gZBGyhNaJk5R_CG25k/pub?gid=0&single=true&output=csv",
        index_col="date",
        parse_dates=True,
    )
    compiled_data = compiled_data.rename(
        columns={"in_hospital": "In hospital confirmed"}
    )

    return compiled_data
