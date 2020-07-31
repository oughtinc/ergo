"""
This module lets you get question and prediction information from PredictIt
and submit predictions, via the API (https://predictit.freshdesk.com/support/solutions/articles/12000001878)

**Example**
In this example, we predict the odds of democrats sweeping the house, senate, and presidency in the 2020 election based on the combined predictit odds.

We submit our prediction to the metaculus question asking "Will Democrats win both halves of Congress + Presidency in the US 2020 Election?"

https://www.metaculus.com/questions/3504/

.. doctest::
    >>> import ergo
    >>> import os
    >>> pi = ergo.PredictIt()
    >>> metaculus = ergo.Metaculus(
    ...     username=os.getenv("METACULUS_USERNAME"),
    ...     password=os.getenv("METACULUS_PASSWORD"),
    ...     api_domain="www"
    ... )

    >>> q_pres_dem = pi.search_market("party wins pres").search_question("dem")
    >>> q_senate_dem = pi.search_market("who control senate").search_question("dem")
    >>> q_house_dem = pi.search_market("who control house").search_question("dem")
    >>> implied = q_pres_dem.get_community_prediction() * q_senate_dem.get_community_prediction() * q_house_dem.get_community_prediction()

    >>> q_sweep = metaculus.get_question(3504)
    >>> q_sweep.submit(implied)
    <Response [202]>
"""
import re
from typing import Any, Dict, Iterator, List

import pandas as pd
import requests
from dateutil.parser import parse
from fuzzywuzzy import fuzz

from ergo.distributions.base import flip


class PredictItQuestion:
    """
    A single binary question in a PredictIt market.

    :param market: PredictIt question instance
    :param data: Contract JSON retrieved from PredictIt API

    :ivar PredictItQuestion question: predictit question instance
    :ivar int id: id of the contract
    :ivar datetime.datetime dateEnd: end-date of a market, usually None
    :ivar str image: url of the image resource for the contract
    :ivar str name: name of the contract
    :ivar str shortName: shortened name of the contract
    :ivar str status: status of the contract. Closed markets aren't included in the API, so always "Open"
    :ivar float lastTradePrice: last price the contract was traded at
    :ivar float bestBuyYesCost: cost to buy a single Yes share
    :ivar float bestBuyNoCost: cost to buy a single No share
    :ivar float bestSellYesCost: cost to sell a single Yes share
    :ivar float bestSellNoCost: cost to sell a single No share
    :ivar float lastClosePrice: price the contract closed at the previous day
    :ivar int displayOrder: position of the contract in PredictIt. Defaults to 0 if sorted by lastTradePrice
    """

    def __init__(self, market: Any, data: Dict):
        self.market = market
        self._data = data

    def __repr__(self):
        if self._data:
            return f'<PredictItQuestion title="{self.name}">'
        else:
            return "<PredictItQuestion>"

    def __str__(self):
        return repr(self)

    def __getattr__(self, name: str):
        """
        If an attribute isn't directly on the class, check whether it's in the
        raw contract data. If it's a time, format it appropriately.
        :param name:
        :return: attribute value
        """
        if name in self._data:
            if name == "dateEnd":
                if self._data[name] == 'N/A':
                    return None
                try:
                    return parse(self._data[name])
                except ValueError:
                    print(
                        f"The column {name} could not be converted into a datetime"
                    )
                    return self._data[name]
            return self._data[name]
        else:
            raise AttributeError(
                f"Attribute {name} is neither directly on this class nor in the raw question data"
            )

    def set_data(self, key: str, value: Any):
        """
        Set key on data dict
        :param key:
        :param value:
        """
        self._data[key] = value

    @staticmethod
    def to_dataframe(
            questions: List["PredictItQuestion"],
            columns: List[str] = ["id", "name", "dateEnd"],
    ) -> pd.DataFrame:
        """
        Summarize a list of questions in a dataframe
        :param questions: questions to summarize
        :param columns: list of column names as strings
        :return: pandas dataframe summarizing the questions
        """

        data = [
            [question._data[key] for key in columns]
            for question in questions
        ]

        return pd.DataFrame(data, columns=columns)

    def get_community_prediction(self):
        """
        Return the lastTradePrice
        """
        return self.lastTradePrice

    def refresh(self):
        """
        Refetch the market data from PredictIt and reload the question.
        """
        self.market.refresh()
        self._data = self.market.get_question_by_id(self.id)._data

    def sample_community(self) -> bool:
        """
        Sample from the PredictIt community distribution (Bernoulli).
        :return: true/false
        """
        return flip(self.get_community_prediction())


class PredictItMarket:
    """
        A PredictIt market.

        :param predictit: PredictIt API instance
        :param data: Market JSON retrieved from PredictIt API

        :ivar PredictIt predictit: predictit api instance
        :ivar str api_url: url of the predictit api for the given question
        :ivar int id: id of the market
        :ivar str name: name of the market
        :ivar str shortName: shortened name of the market
        :ivar str image: url of the image resource of the market
        :ivar str url: url of the market in PredictIt
        :ivar str status: status of the market. Closed markets aren't included in the API, so always "Open"
        :ivar datetime.datetime timestamp: last time the market was updated.
            Api updates every minute, but timestamp can be earlier if it hasn't been traded in
    """

    def __init__(self, predictit: Any, data: Dict):
        self.predictit = predictit
        self._load_attr(data)

    def __repr__(self):
        if self._data:
            return f'<PredictItMarket title="{self.name}">'
        else:
            return "<PredictItMarket>"

    def __str__(self):
        return repr(self)

    def _load_attr(self, data):
        """
        Load attributes of question from data to instance.
        :param data:
        """
        self._data = data
        self.id = data['id']
        self.name = data['name']
        self.shortName = data['shortName']
        self.image = data['image']
        self.url = data['url']
        self.status = data['status']
        self.api_url = f"{self.predictit.api_url}/markets/{self.id}/"
        if data['timeStamp'] == 'N/A':
            self.timestamp = None
        else:
            self.timestamp = parse(data['timeStamp'])

    @property
    def questions(self) -> Iterator[PredictItQuestion]:
        """
        Generate all of the questions in the market.
        :return: iterator of questions in market
        """
        for data in self._data['contracts']:
            yield PredictItQuestion(self, data)

    def refresh(self):
        """
        Refetch the market data from PredictIt,
        used when the question data might have changed.
        """
        r = self.predictit._get(self.api_url)
        self._load_attr(r.json())

    def get_question_by_id(self, id: int) -> PredictItQuestion:
        """
        Return the specified question given by the id number.
        :param id:
        :return: question
        """
        for question in self.questions:
            if question.id == id:
                return question
        raise ValueError("Unable to find a question with that id.")

    def search_question(self, name: str) -> PredictItQuestion:
        """
        Return the specified question given by the name of the question,
        using fuzzy matching in the case where the name isn't exact.
        :param name:
        :return: question
        """
        guess = re.sub(r'[^\w\s]', '', name).lower()
        guess_words = guess.split()
        most_matches = 0
        best_diff = 0
        best_diff_contract = None
        for contract in self.questions:
            short_name = re.sub(r'[^\w\s]', '', contract.name).lower()
            matches = sum([word in short_name for word in guess_words])
            diff = fuzz.token_sort_ratio(guess, short_name)
            if matches > most_matches or (matches >= most_matches and (diff > best_diff)):
                best_diff = diff
                best_diff_contract = contract
            if matches > most_matches:
                most_matches = matches
        if best_diff_contract is None:
            raise ValueError("Unable to find a question with that name.")
        return best_diff_contract


class PredictIt:
    """
    The main class for interacting with PredictIt.
    """

    def __init__(self):
        self.api_url = "https://www.predictit.org/api/marketdata"
        self.s = requests.Session()
        self._data = self._get(f"{self.api_url}/all/").json()

    def _get(self, url):
        """
        Send a get request to to PredictIt API.
        :param url:
        :return: response
        """
        r = self.s.get(url)
        assert "Slow down!" not in str(r.content), "Hit API rate limit"
        return r

    def refresh_markets(self):
        """
        Refetch all of the markets from the predictit api.
        """
        self._data = self._get(f"{self.api_url}/all/").json()

    def get_markets(self) -> Iterator[PredictItMarket]:
        """
        Generate all of the markets currently in PredictIt.
        :return: iterator of predictit markets
        """
        for data in self._data['markets']:
            yield PredictItMarket(self, data)

    def get_market(self, id: int) -> PredictItMarket:
        """
        Return the PredictIt market with the given id.
        A market's id can be found in the url of the market.
        :param id:
        :return: market
        """
        for data in self._data['markets']:
            if data['id'] == id:
                return PredictItMarket(self, data)
        raise ValueError("Unable to find a market with that ID.")

    def search_market(self, name: str) -> PredictItMarket:
        """
        Return a PredictIt market with the given name,
        using fuzzy matching if an exact match is not found.
        :param name:
        :return: market
        """
        return self.get_market(self._get_market_id(name))

    def _get_market_id(self, name: str) -> int:
        """
        Find the id of the market with a given name,
        using fuzzy matching if an exact match is not found.
        :param name:
        :return: market id
        """
        guess = re.sub(r'[^\w\s]', '', name).lower()
        guess_words = guess.split()
        most_matches = 0
        best_diff = 0
        best_diff_id = 0
        for market in self.get_markets():
            short_name = re.sub(r'[^\w\s]', '', market.shortName).lower()
            long_name = re.sub(r'[^\w\s]', '', market.name).lower()
            matches = sum([word in short_name or word in long_name for word in guess_words])
            diff1 = fuzz.token_sort_ratio(guess, short_name)
            diff2 = fuzz.token_sort_ratio(guess, long_name)
            if matches > most_matches or (matches >= most_matches and (diff1 > best_diff or diff2 > best_diff)):
                best_diff = max(diff1, diff2)
                best_diff_id = market.id
            if matches > most_matches:
                most_matches = matches
        return best_diff_id
