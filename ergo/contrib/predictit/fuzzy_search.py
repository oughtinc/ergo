import re

from fuzzywuzzy import fuzz

from ergo import PredictIt, PredictItMarket, PredictItQuestion


def _get_market_id(pi: PredictIt, name: str) -> int:
    guess = re.sub(r"[^\w\s]", "", name).lower()
    guess_words = guess.split()
    most_matches = 0
    best_diff = 0
    best_diff_id = 0
    for market in pi.markets:
        short_name = re.sub(r"[^\w\s]", "", market.shortName).lower()
        long_name = re.sub(r"[^\w\s]", "", market.name).lower()
        matches = sum([word in short_name or word in long_name for word in guess_words])
        diff1 = fuzz.token_sort_ratio(guess, short_name)
        diff2 = fuzz.token_sort_ratio(guess, long_name)
        if matches > most_matches or (
            matches >= most_matches and (diff1 > best_diff or diff2 > best_diff)
        ):
            best_diff = max(diff1, diff2)
            best_diff_id = market.id
        if matches > most_matches:
            most_matches = matches
    return best_diff_id


def search_market(pi: PredictIt, name: str) -> PredictItMarket:
    """
    Return a PredictIt market with the given name,
    using fuzzy matching if an exact match is not found.
    :param pi:
    :param name:
    :return: market
    """
    return pi.get_market(_get_market_id(pi, name))


def search_question(market: PredictItMarket, name: str) -> PredictItQuestion:
    """
    Return the specified question given by the name of the question,
    using fuzzy matching in the case where the name isn't exact.
    :param market:
    :param name:
    :return: question
    """
    guess = re.sub(r"[^\w\s]", "", name).lower()
    guess_words = guess.split()
    most_matches = 0
    best_diff = 0
    best_diff_contract = None
    for contract in market.questions:
        short_name = re.sub(r"[^\w\s]", "", contract.name).lower()
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
