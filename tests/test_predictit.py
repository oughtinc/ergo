import datetime

import pandas as pd

import ergo


def test_refresh(predictit):
    predictit.refresh_markets()
    assert "markets" in predictit._data


def test_markets(predictit):
    markets = list(predictit.markets)[0:10]
    for market in markets:
        assert type(market) is ergo.PredictItMarket
        market_other = predictit.get_market(market.id)
        assert market.name == market_other.name


def test_market_attributes(predictit_markets):
    attrs = {
        "predictit": ergo.PredictIt,
        "api_url": str,
        "id": int,
        "name": str,
        "shortName": str,
        "image": str,
        "url": str,
        "status": str,
        "timeStamp": datetime.datetime,
    }
    for market in predictit_markets:
        for attr in attrs.items():
            assert type(getattr(market, attr[0])) is attr[1]


def test_market_questions(predictit_markets):
    for market in predictit_markets:
        questions = list(market.questions)
        assert len(questions) >= 1
        for question in questions:
            assert type(question) is ergo.PredictItQuestion
            question_other = market.get_question(question.id)
            assert question.name == question_other.name


def test_market_refresh(predictit_markets):
    market = predictit_markets[0]
    name = market.name
    market.refresh()
    assert market.name == name


def test_question_refresh(predictit_markets):
    market = predictit_markets[-1]
    question = list(market.questions)[0]
    name = question.name
    question.refresh()
    assert question.name == name


def test_question_dataframe(predictit_markets):
    for market in predictit_markets:
        questions = list(market.questions)
        df = ergo.PredictItQuestion.to_dataframe(questions)
        assert type(df) is pd.DataFrame


def test_question_attributes(predictit_markets):
    attrs = {
        "market": ergo.PredictItMarket,
        "id": int,
        "name": str,
        "shortName": str,
        "image": str,
        "status": str,
        "displayOrder": int,
    }
    for market in predictit_markets:
        for question in market.questions:
            for attr in attrs.items():
                assert type(getattr(question, attr[0])) is attr[1]
            assert (
                type(getattr(question, "dateEnd")) is datetime.datetime
                or getattr(question, "dateEnd") is None
            )
