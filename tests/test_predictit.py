import datetime

import ergo


def test_market_attributes(predictit_markets):
    """
    Ensure that the PredictIt API hasn't changed.
    This test goes through the various attributes of a market and makes sure they were created.
    """
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


def test_question_attributes(predictit_markets):
    """
    Ensure that the PredictIt API hasn't changed.
    This test goes through the various attributes of a question and makes sure they were created.
    """
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
