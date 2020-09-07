import datetime
from http import HTTPStatus
import pprint

import jax.numpy as np
import pytest
import requests

pp = pprint.PrettyPrinter(indent=4)


def test_get_question(metaculus_questions):
    """Make sure we're getting the user-specific data."""
    assert "my_predictions" in metaculus_questions.continuous_linear_open_question.data


def test_question_name_default(metaculus_questions, log_question_data):
    """Make sure we get the correct default name if no name is specified."""
    assert metaculus_questions.log_question.name == log_question_data["title"]


def test_date_normalize_denormalize(metaculus_questions, date_samples):
    normalized = metaculus_questions.continuous_linear_date_open_question.normalize_samples(
        date_samples
    )
    denormalized = metaculus_questions.continuous_linear_date_open_question.denormalize_samples(
        normalized
    )
    assert (denormalized == date_samples).all()


def test_normalize_denormalize(metaculus_questions):
    samples = np.array([0, 0.5, 1, 5, 10, 20])
    normalized = metaculus_questions.log_question.normalize_samples(samples)
    denormalized = metaculus_questions.log_question.denormalize_samples(normalized)
    assert denormalized == pytest.approx(samples, abs=1e-5)


def test_submit_continuous_linear_open(metaculus_questions, logistic_mixture):
    submission = metaculus_questions.continuous_linear_open_question.prepare_logistic_mixture(
        logistic_mixture
    )
    r = metaculus_questions.continuous_linear_open_question.submit(submission)
    assert r.status_code == HTTPStatus.ACCEPTED


def test_submit_continuous_linear_date_open(
    metaculus_questions, normalized_logistic_mixture
):
    submission = metaculus_questions.continuous_linear_date_open_question.prepare_logistic_mixture(
        normalized_logistic_mixture
    )
    r = metaculus_questions.continuous_linear_date_open_question.submit(submission)
    assert r.status_code == HTTPStatus.ACCEPTED


def test_submit_continuous_linear_closed(
    metaculus_questions, normalized_logistic_mixture
):
    submission = metaculus_questions.continuous_linear_closed_question.prepare_logistic_mixture(
        normalized_logistic_mixture
    )
    r = metaculus_questions.continuous_linear_closed_question.submit(submission)
    assert r.status_code == 202


def test_submit_continuous_log_open(metaculus_questions, normalized_logistic_mixture):
    submission = metaculus_questions.continuous_log_open_question.prepare_logistic_mixture(
        normalized_logistic_mixture
    )
    r = metaculus_questions.continuous_log_open_question.submit(submission)
    assert r.status_code == 202


def test_submit_from_samples(metaculus_questions, logistic_mixture_samples):
    r = metaculus_questions.continuous_linear_open_question.submit_from_samples(
        logistic_mixture_samples
    )
    assert r.status_code == 202


def test_submit_binary(metaculus_questions):
    r = metaculus_questions.binary_question.submit(0.95)
    assert r.status_code == 202


def test_submit_closed_question_fails(metaculus_questions, normalized_logistic_mixture):
    with pytest.raises(requests.exceptions.HTTPError):
        submission = metaculus_questions.closed_question.prepare_logistic_mixture(
            normalized_logistic_mixture
        )
        metaculus_questions.closed_question.submit(submission)


def test_score_binary(metaculus_questions):
    """Smoke test"""
    metaculus_questions.binary_question.score_my_predictions()


def test_get_questions(metaculus):
    questions = metaculus.get_questions(question_status="closed")
    assert len(questions) >= 15


def test_get_questions_json(metaculus):
    questions = metaculus.get_questions_json(include_discussion_questions=True)
    assert len(questions) >= 20


def test_get_questions_json_pages(metaculus):
    two_pages = metaculus.get_questions_json(pages=2, include_discussion_questions=True)
    assert len(two_pages) >= 40


def test_get_questions_player_status(metaculus):
    qs_i_predicted = metaculus.make_questions_df(
        metaculus.get_questions_json(player_status="predicted")
    )
    assert qs_i_predicted["i_predicted"].all()

    not_predicted = metaculus.make_questions_df(
        metaculus.get_questions_json(player_status="not-predicted")
    )
    assert (not_predicted["i_predicted"] == False).all()  # noqa: E712


def test_get_questions_question_status(metaculus):
    open = metaculus.make_questions_df(
        metaculus.get_questions_json(question_status="open")
    )

    # the additional day is to account for difference in timezones
    assert (
        open["close_time"] > (datetime.datetime.now() - datetime.timedelta(days=1))
    ).all()

    closed = metaculus.make_questions_df(
        metaculus.get_questions_json(question_status="closed")
    )
    assert (
        closed["close_time"] < (datetime.datetime.now() + datetime.timedelta(days=1))
    ).all()


# @pytest.mark.xfail(reason="Fitting doesn't reliably work yet #219")
def test_submission_from_samples_linear(metaculus_questions, logistic_mixture_samples):
    normalized_mixture = metaculus_questions.continuous_linear_open_question.get_submission_from_samples(
        logistic_mixture_samples
    )
    normalized_mixture_samples = [normalized_mixture.sample() for _ in range(5000)]
    mixture_samples = metaculus_questions.continuous_linear_open_question.denormalize_samples(
        normalized_mixture_samples
    )
    assert float(np.mean(logistic_mixture_samples)) == pytest.approx(
        float(np.mean(mixture_samples)), rel=0.1
    )
    assert float(np.var(logistic_mixture_samples)) == pytest.approx(
        float(np.var(mixture_samples)), rel=0.2
    )


# @pytest.mark.xfail(reason="Fitting doesn't reliably work yet #219")
def test_submitted_equals_predicted_linear(
    metaculus_questions, logistic_mixture_samples
):
    metaculus_questions.continuous_linear_open_question.submit_from_samples(
        logistic_mixture_samples
    )
    metaculus_questions.continuous_linear_open_question.refresh_question()
    latest_prediction = (
        metaculus_questions.continuous_linear_open_question.get_latest_normalized_prediction()
    )
    true_mixture = metaculus_questions.continuous_linear_open_question.get_true_scale_mixture(
        latest_prediction
    )
    prediction_samples = np.array([true_mixture.sample() for _ in range(0, 1000)])
    assert float(np.mean(logistic_mixture_samples)) == pytest.approx(
        float(np.mean(prediction_samples)), rel=0.1
    )


# @pytest.mark.xfail(reason="Fitting doesn't reliably work yet #219")
def test_submitted_equals_predicted_log(metaculus_questions, logistic_mixture_samples):
    metaculus_questions.continuous_log_open_question.submit_from_samples(
        logistic_mixture_samples
    )
    metaculus_questions.continuous_log_open_question.refresh_question()
    latest_prediction = (
        metaculus_questions.continuous_log_open_question.get_latest_normalized_prediction()
    )
    normalized_prediction_samples = np.array(
        [latest_prediction.sample() for _ in range(0, 5000)]
    )
    prediction_samples = metaculus_questions.continuous_log_open_question.denormalize_samples(
        normalized_prediction_samples
    )
    assert float(np.mean(logistic_mixture_samples)) == pytest.approx(
        float(np.mean(prediction_samples)), rel=0.1
    )


def test_get_community_prediction_linear(metaculus_questions):
    assert metaculus_questions.continuous_linear_closed_question.sample_community() > 0


def test_get_community_prediction_log(metaculus_questions):
    assert metaculus_questions.continuous_log_open_question.sample_community() > 0


def test_sample_community_binary(metaculus_questions):
    value = metaculus_questions.binary_question.sample_community()
    assert bool(value) in (True, False)
