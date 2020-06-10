import math
import os
from types import SimpleNamespace
from typing import cast

from dotenv import load_dotenv
import jax.numpy as np
import pandas as pd
import pytest

import ergo

from ergo.distributions import Logistic, LogisticMixture, TruncatedLogisticMixture
from ergo.scale import Scale


def three_sd_scale(loc, s):
    sd = s * math.pi / math.sqrt(3)
    return Scale(loc - 3 * sd, loc + 3 * sd)


def easyLogistic(loc, scale):
    return Logistic(loc, scale, three_sd_scale(loc, scale))


@pytest.fixture(scope="module")
def logistic_mixture10():
    xscale = Scale(-20, 40)
    return LogisticMixture(
        components=[
            Logistic(loc=15, s=2.3658268, scale=xscale),
            Logistic(loc=5, s=2.3658268, scale=xscale),
        ],
        probs=[0.5, 0.5],
        scale=xscale,
    )


@pytest.fixture(scope="module")
def logistic_mixture_p_uneven():
    xscale = Scale(-10, 20)
    return LogisticMixture(
        components=[
            Logistic(loc=10, s=3, scale=xscale),
            Logistic(loc=5, s=5, scale=xscale),
        ],
        probs=[1.8629593e-29, 1.0],
        scale=xscale,
    )


@pytest.fixture(scope="module")
def truncated_logistic_mixture():
    xscale = Scale(5000, 120000)
    return TruncatedLogisticMixture(
        components=[Logistic(loc=10000, s=1000, scale=xscale), Logistic(loc=100000, s=10000, scale=xscale)],
        probs=[0.8, 0.2],
        floor=5000,
        ceiling=500000,
        scale=xscale
    )


@pytest.fixture(scope="module")
def logistic_mixture_samples(logistic_mixture, n=1000):
    return np.array([logistic_mixture.sample() for _ in range(0, n)])


@pytest.fixture(scope="module")
def logistic_mixture_p_overlapping():
    xscale = three_sd_scale(4000000.035555004, 200000.02)
    return LogisticMixture(
        components=[
            Logistic(4000000.035555004, 200000.02, xscale),
            Logistic(4000000.0329152746, 200000.0, xscale),
        ],
        probs=[0.5, 0.5],
        scale=xscale,
    )


@pytest.fixture(scope="module")
def logistic_mixture_norm_test():
    xscale = Scale(-50, 50)
    return LogisticMixture(
        components=[Logistic(-40, 1, xscale), Logistic(50, 10, xscale)],
        probs=[0.5, 0.5],
        scale=xscale,
    )


@pytest.fixture(scope="module")
def logistic_mixture15():
    xscale = Scale(-10, 40)
    return LogisticMixture(
        components=[
            Logistic(loc=10, s=3.658268, scale=xscale),
            Logistic(loc=20, s=3.658268, scale=xscale),
        ],
        probs=[0.5, 0.5],
        scale=xscale,
    )


@pytest.fixture(scope="module")
def logistic_mixture_samples(logistic_mixture, n=1000):
    return np.array([logistic_mixture.sample() for _ in range(0, n)])


@pytest.fixture(scope="module")
def log_question_data():
    return {
        "id": 0,
        "possibilities": {
            "type": "continuous",
            "scale": {"deriv_ratio": 10, "min": 1, "max": 10},
        },
        "title": "question_title",
    }


@pytest.fixture(scope="module")
def metaculus():
    load_dotenv()
    uname = cast(str, os.getenv("METACULUS_USERNAME"))
    pwd = cast(str, os.getenv("METACULUS_PASSWORD"))
    user_id_str = cast(str, os.getenv("METACULUS_USER_ID"))
    if None in [uname, pwd, user_id_str]:
        raise ValueError(
            ".env is missing METACULUS_USERNAME, METACULUS_PASSWORD, or METACULUS_USER_ID"
        )
    user_id = int(user_id_str)
    metaculus = ergo.Metaculus(uname, pwd)
    assert metaculus.user_id == user_id
    return metaculus


@pytest.fixture(scope="module")
def metaculus_questions(metaculus, log_question_data):
    questions = SimpleNamespace()
    questions.continuous_linear_closed_question = metaculus.get_question(3963)
    questions.continuous_linear_open_question = metaculus.get_question(3962)
    questions.continuous_linear_date_open_question = metaculus.get_question(4212)
    questions.continuous_log_open_question = metaculus.get_question(3961)
    questions.closed_question = metaculus.get_question(3965)
    questions.binary_question = metaculus.get_question(3966)
    questions.log_question = metaculus.make_question_from_data(log_question_data)
    return questions


@pytest.fixture(scope="module")
def date_samples(metaculus_questions, normalized_logistic_mixture):
    return metaculus_questions.continuous_linear_date_open_question.denormalize_samples(
        pd.Series([normalized_logistic_mixture.sample() for _ in range(0, 1000)])
    )


@pytest.fixture(scope="module")
def histogram():
    return make_histogram()


def make_histogram():
    xs = np.array(
        [
            -0.22231131421566422,
            0.2333153619512007,
            0.6889420381180656,
            1.1445687142849306,
            1.6001953904517954,
            2.0558220666186604,
            2.5114487427855257,
            2.9670754189523905,
        ]
    )
    densities = np.array(
        [
            0.05020944540593859,
            0.3902426887736647,
            0.5887675161478794,
            0.19516571803813396,
            0.33712516238248535,
            0.4151935926066581,
            0.16147625748938946,
            0.03650993407810862,
        ]
    )
    return {"xs": xs, "densities": densities}
