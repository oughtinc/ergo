"""
This module lets you get question and prediction information from PredictIt
and submit predictions, via the API (https://predictit.freshdesk.com/support/solutions/articles/12000001878)

**Example**
In this example, we grap the oldest PredictIt question and it's first contract and check to make sure the name's of both are consistent while we refresh.

.. doctest::
    >>> import ergo
    >>> import os
    >>> pi = ergo.PredictIt()

    >>> market = list(pi.markets)[0]
    >>> contract = list(market.questions)[0]

    >>> name1 = market.name
    >>> print(type(name1) is str)
    True

    >>> market.refresh()

    >>> name2 = market.name
    >>> print(name2 == name1)
    True

    >>> contract.refresh()

    >>> name3 = contract.market.name
    >>> print(name3 == name1)
    True
"""
from typing import Any, Dict, Generator, List

from dateutil.parser import parse
import pandas as pd
import requests

from ergo.distributions.base import flip


class PredictItQuestion:
    """
    A single binary question in a PredictIt market.

    :param market: PredictIt question instance
    :param data: Contract JSON retrieved from PredictIt API

    :ivar PredictItQuestion market: predictit market instance
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

    def __init__(self, market: "PredictItMarket", data: Dict):
        self.market = market
        self._data = data

    def __repr__(self):
        return f'<PredictItQuestion title="{self.name}">'

    def __getattr__(self, name: str):
        """
        If an attribute isn't directly on the class, check whether it's in the
        raw contract data. If it's a time, format it appropriately.
        :param name:
        :return: attribute value
        """
        if name not in self._data:
            raise AttributeError(
                f"Attribute {name} is neither directly on this class nor in the raw question data"
            )
        if name != "dateEnd":
            return self._data[name]
        dateEnd = self._data["dateEnd"]
        if dateEnd == "N/A":
            return None
        try:
            return parse(dateEnd)
        except ValueError:
            print(f"The column {name} could not be converted into a datetime")
            return dateEnd

    def set_data(self, key: str, value: Any):
        """
        Set key on data dict
        :param key:
        :param value:
        """
        self._data[key] = value

    @staticmethod
    def to_dataframe(
        questions: List["PredictItQuestion"], columns=None,
    ) -> pd.DataFrame:
        """
        Summarize a list of questions in a dataframe
        :param questions: questions to summarize
        :param columns: list of column names as strings
        :return: pandas dataframe summarizing the questions
        """
        if columns is None:
            columns = ["id", "name", "dateEnd"]
        data = [[question._data[key] for key in columns] for question in questions]

        return pd.DataFrame(data, columns=columns)

    def get_community_prediction(self) -> float:
        """
        Return the lastTradePrice
        """
        return self.lastTradePrice

    def refresh(self):
        """
        Refetch the market data from PredictIt and reload the question.
        """
        self.market.refresh()
        self._data = self.market.get_question(self.id)._data

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

    def __init__(self, predictit: "PredictIt", data: Dict):
        self.predictit = predictit
        self._load_attr(data)

    def _get(self, url: str) -> requests.Response:
        """
        Send a get request to to PredictIt API.
        :param url:
        :return: response
        """
        r = self.predictit.s.get(url)
        if "Slow down!" in str(r.content):
            raise requests.RequestException("Hit API rate limit")
        return r

    def __repr__(self):
        return f'<PredictItMarket title="{self.name}">'

    def _load_attr(self, data):
        """
        Load attributes of question from data to instance.
        :param data:
        """
        self._data = data
        self.id = data["id"]
        self.name = data["name"]
        self.shortName = data["shortName"]
        self.image = data["image"]
        self.url = data["url"]
        self.status = data["status"]
        self.api_url = f"{self.predictit.api_url}/markets/{self.id}/"
        if data["timeStamp"] == "N/A":
            self.timestamp = None
        else:
            self.timestamp = parse(data["timeStamp"])

    @property
    def questions(self) -> Generator[PredictItQuestion, None, None]:
        """
        Generate all of the questions in the market.
        :return: iterator of questions in market
        """
        for data in self._data["contracts"]:
            yield PredictItQuestion(self, data)

    def refresh(self):
        """
        Refetch the market data from PredictIt,
        used when the question data might have changed.
        """
        r = self._get(self.api_url)
        self._load_attr(r.json())

    def get_question(self, id: int) -> PredictItQuestion:
        """
        Return the specified question given by the id number.
        :param id:
        :return: question
        """
        for question in self.questions:
            if question.id == id:
                return question
        raise ValueError("Unable to find a question with that id.")


class PredictIt:
    """
    The main class for interacting with PredictIt.
    """

    def __init__(self):
        self.api_url = "https://www.predictit.org/api/marketdata"
        self.s = requests.Session()
        self._data = self._get(f"{self.api_url}/all/").json()

    def _get(self, url: str) -> requests.Response:
        """
        Send a get request to to PredictIt API.
        :param url:
        :return: response
        """
        r = self.s.get(url)
        if "Slow down!" in str(r.content):
            raise requests.RequestException("Hit API rate limit")
        return r

    def refresh_markets(self):
        """
        Refetch all of the markets from the predictit api.
        """
        self._data = self._get(f"{self.api_url}/all/").json()

    @property
    def markets(self) -> Generator[PredictItMarket, None, None]:
        """
        Generate all of the markets currently in PredictIt.
        :return: iterator of predictit markets
        """
        for data in self._data["markets"]:
            yield PredictItMarket(self, data)

    def get_market(self, id: int) -> PredictItMarket:
        """
        Return the PredictIt market with the given id.
        A market's id can be found in the url of the market.
        :param id:
        :return: market
        """
        for data in self._data["markets"]:
            if data["id"] == id:
                return PredictItMarket(self, data)
        raise ValueError("Unable to find a market with that ID.")
