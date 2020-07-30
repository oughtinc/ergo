"""
This module lets you get question and prediction information from PredictIt
and submit predictions, via the API (https://predictit.freshdesk.com/support/solutions/articles/12000001878)
"""
import re
from typing import Any, Dict, Iterator

import requests
from dateutil.parser import parse
from fuzzywuzzy import fuzz


class Contract:
    """
    A single contract in a PredictIt market.

    :param question: PredictIt question instance
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

    def __init__(self, question, data):
        self.question = question
        self._data = data

    def __getattr__(self, name):
        """
        If an attribute isn't directly on the class, check whether it's in the
        raw contract data. If it's a time, format it appropriately.
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


class PredictItQuestion:
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
        self.api_url = f"{predictit.api_url}/markets/{id}/"
        self._load_attr(data)

    def _load_attr(self, data):
        """
        Load attributes of question from data to instance.
        """
        self._data = data
        self.id = data['id']
        self.name = data['name']
        self.shortName = data['shortName']
        self.image = data['image']
        self.url = data['url']
        self.status = data['status']
        if data['timeStamp'] == 'N/A':
            self.timestamp = None
        else:
            self.timestamp = parse(data['timeStamp'])

    @property
    def contracts(self) -> Iterator[Contract]:
        """
        Generate all of the contracts in the market.
        """
        for data in self._data['contracts']:
            yield Contract(self, data)

    def refresh_question(self):
        """
        Refetch the question data from PredictIt,
        used when the question data might have changed.
        """
        r = self.predictit.s.get(self.api_url)
        self._load_attr(r.json())

    def get_contract_bin(self, bin_num: int) -> Contract:
        """
        Return the specified contract given by the bin number, starting at 0.
        """
        return list(self.contracts)[bin_num]

    def search_contract(self, bin_name: str) -> Contract:
        """
        Return the specified contract given by the name of the contract,
        using fuzzy matching in the case where the name isn't exact.
        """
        guess = re.sub(r'[^\w\s]', '', bin_name).lower()
        guess_words = guess.split()
        most_matches = 0
        best_diff = 0
        best_diff_contract = 0
        for contract in self.contracts:
            short_name = re.sub(r'[^\w\s]', '', contract.name).lower()
            matches = sum([word in short_name for word in guess_words])
            diff = fuzz.token_sort_ratio(guess, short_name)
            if matches > most_matches or (matches >= most_matches and (diff > best_diff)):
                best_diff = diff
                best_diff_contract = contract
            if matches > most_matches:
                most_matches = matches
        return best_diff_contract


class PredictIt:
    """
    The main class for interacting with PredictIt.
    """

    def __init__(self):
        self.api_url = "https://www.predictit.org/api/marketdata"
        self.s = requests.Session()

    def get_questions(self) -> Iterator[PredictItQuestion]:
        """
        Generate all of the questions currently in PredictIt.
        """
        r = self.s.get(f"{self.api_url}/all/")
        for data in r.json()['markets']:
            yield PredictItQuestion(self, data)

    def get_question(self, id: int) -> PredictItQuestion:
        """
        Return the PredictIt question with the given id.
        A question's id can be found in the url of the question.
        """
        r = self.s.get(f"{self.api_url}/markets/{id}/")
        data = r.json()
        return PredictItQuestion(self, data)

    def search_question(self, name: str) -> PredictItQuestion:
        """
        Return a PredictIt question with the given name,
        using fuzzy matching if an exact match is not found.
        """
        id = self._get_market_id(name)
        markets = self.get_questions()
        for market in markets:
            if market.id == id:
                return market

    def _get_market_id(self, market_str: str) -> int:
        """
        Find the id of the market with a given name,
        using fuzzy matching if an exact match is not found.
        """
        guess = re.sub(r'[^\w\s]', '', market_str).lower()
        guess_words = guess.split()
        most_matches = 0
        best_diff = 0
        best_diff_id = 0
        for market in self.get_questions():
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


p = PredictIt()
q = p.get_question(2721)
print(q.contracts.__class__)
print(q.timestamp)
print(q.search_contract("Lib").shortName)
print(p.search_question("2020 pres").name)
