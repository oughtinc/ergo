"""
This module accesses market data from the prediction site PredictIt

**Example**

"""

import bisect
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import functools
import json
import math
import textwrap
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from plotnine import (  # type: ignore
    aes,
    element_text,
    facet_wrap,
    geom_density,
    geom_histogram,
    ggplot,
    guides,
    labs,
    scale_fill_brewer,
    scale_x_continuous,
    scale_x_datetime,
    scale_x_log10,
    theme,
    xlim,
)

import pyro.distributions as dist
import requests
from scipy import stats
import torch
from typing_extensions import Literal

import ergo.logistic as logistic
import ergo.ppl as ppl
from ergo.theme import ergo_theme  # type: ignore

class Contract:
    attribute_map = {
        'contract_id': 'id',
        'date_end': 'dateEnd',
        'name': 'shortName',
        'status': 'status',
        'last_trade_price': 'lastTradePrice',
        'best_buy_yes': 'bestBuyYesCost',
        'best_buy_no': 'bestBuyNoCost',
        'best_sell_yes': 'bestSellYesCost',
        'best_sell_no': 'bestSellNoCost',
        'last_close_price': 'lastClosePrice'}

    def __init__(self, contract_dict: Dict):
        for attr in self.attribute_map:
            setattr(self, attr, contract_dict[self.attribute_map[attr]])
        
class Market:
    
    def __init__(self, market_dict: Dict):
        self.market_id = market_dict['id']
        self.name = market_dict['shortName']
        self.url = market_dict['url']
        self.status = market_dict['status']
        self.timestamp = market_dict['timestamp']
        self.contracts = self.read_contracts(market_dict['contracts'])

    def read_contracts(self, contract_dicts: List[Dict]):
        for contract_dict in contract_dicts:
            contract = Contract(contract_dict)


    def modelmodelmodel(self):
        pass

class PredictIt:
    """
    The main class for interacting with PredictIt

    :param username: A Metaculus username
    :param password: The password for the given Metaculus username
    :param api_domain: A Metaculus subdomain (e.g., www, pandemic, finance)
    """

    predictit_url = 'https://predictit.org/api/marketdata/{market}'
    profit_fee = 0.10

    def __init__(self)
        self.markets = dict()

    def get_market_all(self):
        api_url = predictit_url.format(market = "all/")

        response = requests.get(api_url)
        if response.status_code == requests.codes.ok:
            data = json.loads(response.content.decode('utf-8'))
            if not data.get('markets'):
                raise ValueError(
                    "No Markets in PredictIt response"
                )
            self.markets = self.make_markets_from_data(data['markets'])
        else:
            response.raise_for_status() 

    def get_market(self, market_id: int)
        api_url = predictit_url.format(market = market_id)
        response = requests.get(api_url)
        if response.status_code == requests.codes.ok:
            market_dict = json.loads(response.content.decode('utf-8'))
            market = Market(market_dict)
            self.markets[market] = market
        else:
            response.raise_for_status() 

    def make_markets(self, market_dicts: List[Dict]):
        markets = []
        for market_dict in market_dicts:
            market = Market(market_dict)
            if market:
                markets.append(market)
        return markets

    def market_search(self):
        if not self.markets:
            pass

    def find_negative_risk(self):
        """Identify markets where negative risk strategies can be implemented
        Buying No on all multicontract trades with lots of liquidity can somtimes
        lead to guaranteed profits even when fees are taken into account
        """
        for market in self.markets:
            if len(market.contracts) <= 1:
                continue
            pass
             
