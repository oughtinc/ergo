import operator
import re
from typing import List, Tuple

from fuzzywuzzy import fuzz

from ergo import PredictIt, PredictItMarket, PredictItQuestion


def _get_name_matches(name: str, guess_words: List[str]) -> int:
    """
    Return the number of common words in a str and list of words
    :param name:
    :param guess_words:
    :return: number of matches
    """
    matches = sum(word in name for word in guess_words)
    return matches


def _get_name_score(names: List[str], guess: str) -> int:
    """
    Return similarity score of a guess to a name. Higher is better.
    :param names:
    :param guess:
    :return: score
    """
    names = [re.sub(r"[^\w\s]", "", name).lower() for name in names]
    guess_words = guess.split()
    matches = max(_get_name_matches(name, guess_words) for name in names)
    diff = max(fuzz.token_sort_ratio(guess, name) for name in names)
    return matches * 100 + diff


def _check_market(market: PredictItMarket, guess: str) -> Tuple[int, int]:
    """
    Return the id and similarity score of a market to a guess.
    :param market:
    :param guess:
    :return: id and similarity score
    """
    return market.id, _get_name_score([market.shortName, market.name], guess)


def _check_question(question: PredictItQuestion, guess: str) -> Tuple[int, int]:
    """
    Return the id and similarity score of a question to a guess.
    :param question:
    :param guess:
    :return: id and similarity score
    """
    return question.id, _get_name_score([question.name], guess)


def _get_best_market_id(pi: PredictIt, guess: str) -> int:
    """
    Return the id of the market with the highest similarity score.
    :param pi:
    :param guess:
    :return: market id
    """
    return max(
        (_check_market(market, guess) for market in pi.markets),
        key=operator.itemgetter(1),
    )[0]


def _get_best_question_id(market: PredictItMarket, guess: str) -> int:
    """
    Return the id of the question with the highest similarity score.
    :param market:
    :param guess:
    :return: question id
    """
    return max(
        (_check_question(question, guess) for question in market.questions),
        key=operator.itemgetter(1),
    )[0]


def search_market(pi: PredictIt, guess: str) -> PredictItMarket:
    """
    Return a PredictIt market with the given name,
    using fuzzy matching if an exact match is not found.
    :param pi:
    :param guess:
    :return: market
    """
    return pi.get_market(_get_best_market_id(pi, guess))


def search_question(market: PredictItMarket, guess: str) -> PredictItQuestion:
    """
    Return the specified question given by the name of the question,
    using fuzzy matching in the case where the name isn't exact.
    :param market:
    :param guess:
    :return: question
    """
    return market.get_question(_get_best_question_id(market, guess))
