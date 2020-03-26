import functools
import country_converter
import pandas as pd


class DataLoader:

    def __init__(self):
        self._warnings = set()
        self.data = {}
        self.load()

    def warn(self, warning):
        if not warning in self._warnings:
            print(f"WARNING: {warning}")
        self._warnings.add(warning)

    def get(self, name):
        return self.data[name]

    def load(self):
        pass

    def raw(self):
        return self.data

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)


class WHORegionData(DataLoader):
    url = "https://gist.githubusercontent.com/brachbach/de74ec9fdee315234976084599a9539c/raw/5e12ee7ba283dc0f11d162893984c620b6d9f703/who_regions.csv"

    def load(self):
        data = pd.read_csv(self.url)
        standard_countries = country_converter.convert(
            names=data["Country"].tolist(), to='name_short')
        data['standard_country'] = standard_countries
        self.data = data


class HopkinsData(DataLoader):
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

    def load(self):
        data = pd.read_csv(self.url)
        standard_countries = country_converter.convert(
            names=data["Country/Region"].tolist(), to='name_short')
        data['standard_country'] = standard_countries
        self.data = data


class ConfirmedInfections(DataLoader):
    def load(self):
        self.who_regions = WHORegionData().raw()
        self.hopkins_data = HopkinsData().raw()
        self.countries = set(self.hopkins_data["standard_country"].tolist())
        self.regions = set(self.who_regions["Region"].tolist())

    def countries_for_region(self, region):
        return self.who_regions.loc[self.who_regions["Region"] == region, "standard_country"].tolist()

    @functools.lru_cache(None)
    def confirmed_for_country(self, country, date, warn=False):
        country_data = self.hopkins_data.loc[self.hopkins_data["standard_country"] == country][date.format(
            "M/D/YY")]
        if len(country_data) == 0 and warn:
            self.warn(f"No confirmed case data for country: {country}.")
        return sum(country_data)

    @functools.lru_cache(None)
    def confirmed_for_region(self, region, date):
        countries = self.countries_for_region(region)
        return sum([self.confirmed_for_country(country, date) for country in countries])

    def is_country(self, area):
        return area in self.countries

    def is_region(self, area):
        return area in self.regions

    def get(self, area, date):
        if self.is_country(area):
            return self.confirmed_for_country(area, date)
        elif self.is_region(area):
            return self.confirmed_for_region(area, date)
        else:
            self.warn(f"No confirmed case data for {area}")
            return 0
