Loading Data from Google Sheets
===============================
Three methods for loading data from google sheets into a Colab Notebook


Method 1 (Public CSV)
---------------------
If you're willing to make your spreadsheet public, you can publish it as a CSV file on Google Sheets. Go to File > Publish to the Web, and select the CSV format. Then you can copy the published url, and load it in python using pandas.

.. code-block:: python

    import pandas as pd
    df = pd.read_csv(url)

Method 2 (OAuth)
----------------
This method requires the user of the colab to authorize it every time the colab runs, but can work with non-public sheets

.. code-block:: python

    # Authentication
    import google
    google.colab.auth.authenticate_user()
    google_sheets_credentials = GoogleCredentials.get_application_default()
    gc = gspread.authorize(google_sheets_credentials)

    # Load spreadsheet
    wb = gc.open_by_url(url)
    sheet = wb.worksheet(sheet)
    values = sheet.get_all_values()

Method 3 (Service Account)
--------------------------
This method requires your to follow the instructions at https://gspread.readthedocs.io/en/latest/oauth2.html to create a google service account. You then need to share the google sheet with the service account email address.

.. code-block:: 

    # Need a newer version of gspread than included by default in Colab
    !pip install --upgrade gspread

    service_account_info = {} #JSON for google service account
    import gspread
    from google.oauth2.service_account import Credentials

    scope = ['https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive']

    credentials = Credentials.from_service_account_info(service_account_info, scopes=scope)

    gc = gspread.authorize(credentials)

    # Load spreadsheet
    wb = gc.open_by_url(url)
    sheet = wb.worksheet(sheet)
    values = sheet.get_all_values()
