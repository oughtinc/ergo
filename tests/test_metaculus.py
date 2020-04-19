from datetime import datetime
from http import HTTPStatus
import pprint

import numpy as np
import pandas as pd
import pytest
import requests

import ergo
import tests.mocks

pp = pprint.PrettyPrinter(indent=4)

uname = "oughttest"
pwd = "6vCo39Mz^rrb"
user_id = 112420


class TestMetaculus:
    metaculus = ergo.Metaculus(uname, pwd)
    continuous_linear_closed_question = metaculus.get_question(3963)
    continuous_linear_open_question = metaculus.get_question(3962)
    continuous_linear_date_open_question = metaculus.get_question(4212)
    continuous_log_open_question = metaculus.get_question(3961)
    closed_question = metaculus.get_question(3965)
    binary_question = metaculus.get_question(3966)

    mock_samples = np.array(
        [
            ergo.logistic.sample_mixture(tests.mocks.mock_true_params)
            for _ in range(0, 1000)
        ]
    )

    mock_date_samples = continuous_linear_date_open_question.denormalize_samples(
        pd.Series(
            [
                ergo.logistic.sample_mixture(tests.mocks.mock_normalized_params)
                for _ in range(0, 1000)
            ]
        )
    )

    mock_log_question = metaculus.make_question_from_data(
        tests.mocks.mock_log_question_data
    )

    def test_login(self):
        assert self.metaculus.user_id == user_id

    def test_get_question(self):
        """make sure we're getting the user-specific data"""
        assert "my_predictions" in self.continuous_linear_open_question.data

    def test_date_normalize_denormalize(self):
        samples = self.mock_date_samples
        normalized = self.continuous_linear_date_open_question.normalize_samples(
            samples
        )
        denormalized = self.continuous_linear_date_open_question.denormalize_samples(
            normalized
        )
        assert all(denormalized == samples)

    def test_normalize_denormalize(self):
        samples = [0, 0.5, 1, 5, 10, 20]
        normalized = self.mock_log_question.normalize_samples(samples)
        denormalized = self.mock_log_question.denormalize_samples(normalized)
        assert denormalized == pytest.approx(samples, abs=1e-5)

    def test_submit_continuous_linear_open(self):
        submission = self.continuous_linear_open_question.get_submission(
            tests.mocks.mock_normalized_params
        )
        r = self.continuous_linear_open_question.submit(submission)
        assert r.status_code == HTTPStatus.ACCEPTED

    def test_submit_continuous_linear_date_open(self):
        submission = self.continuous_linear_date_open_question.get_submission(
            tests.mocks.mock_normalized_params
        )
        r = self.continuous_linear_date_open_question.submit(submission)
        assert r.status_code == HTTPStatus.ACCEPTED

    def test_submit_continuous_linear_closed(self):
        submission = self.continuous_linear_closed_question.get_submission(
            tests.mocks.mock_normalized_params
        )
        r = self.continuous_linear_closed_question.submit(submission)
        assert r.status_code == 202

    def test_submit_continuous_log_open(self):
        submission = self.continuous_log_open_question.get_submission(
            tests.mocks.mock_normalized_params
        )
        r = self.continuous_log_open_question.submit(submission)
        assert r.status_code == 202

    def test_submit_from_samples(self):
        r = self.continuous_linear_open_question.submit_from_samples(
            self.mock_samples, samples_for_fit=1000
        )
        assert r.status_code == 202

    def test_submit_binary(self):
        r = self.binary_question.submit(0.95)
        assert r.status_code == 202

    def test_submit_closed_question_fails(self):
        with pytest.raises(requests.exceptions.HTTPError):
            submission = self.closed_question.get_submission(
                tests.mocks.mock_normalized_params
            )
            r = self.closed_question.submit(submission)
            print(r)

    def test_score_binary(self):
        """smoke test"""
        self.binary_question.score_my_predictions()

    def test_get_questions(self):
        questions = self.metaculus.get_questions(question_status="closed")
        assert len(questions) >= 20

    def test_get_questions_json(self):
        questions = self.metaculus.get_questions_json(include_discussion_questions=True)
        assert len(questions) >= 20

    def test_get_questions_json_pages(self):
        two_pages = self.metaculus.get_questions_json(pages=2, include_discussion_questions=True)
        assert len(two_pages) >= 40

    def test_get_questions_player_status(self):
        qs_i_predicted = self.metaculus.make_questions_df(
            self.metaculus.get_questions_json(player_status="predicted")
        )
        assert qs_i_predicted["i_predicted"].all()

        not_predicted = self.metaculus.make_questions_df(
            self.metaculus.get_questions_json(player_status="not-predicted")
        )
        assert (not_predicted["i_predicted"] == False).all()  # noqa: E712

    def test_get_questions_question_status(self):
        open = self.metaculus.make_questions_df(
            self.metaculus.get_questions_json(question_status="open")
        )
        assert (open["close_time"] > datetime.now()).all()

        closed = self.metaculus.make_questions_df(
            self.metaculus.get_questions_json(question_status="closed")
        )
        assert (closed["close_time"] < datetime.now()).all()

    def test_submitted_equals_predicted_linear(self):
        self.continuous_linear_open_question.submit_from_samples(self.mock_samples)
        latest_prediction = (
            self.continuous_linear_open_question.get_latest_normalized_prediction()
        )
        scaled_params = self.continuous_linear_open_question.get_true_scale_mixture(
            latest_prediction
        )
        prediction_samples = np.array(
            [ergo.logistic.sample_mixture(scaled_params) for _ in range(0, 1000)]
        )

        assert np.mean(self.mock_samples) == pytest.approx(
            np.mean(prediction_samples), np.mean(prediction_samples) / 10
        )

    def test_submitted_equals_predicted_log(self):
        self.continuous_log_open_question.submit_from_samples(self.mock_samples)
        latest_prediction = (
            self.continuous_log_open_question.get_latest_normalized_prediction()
        )
        prediction_samples = np.array(
            [
                self.continuous_log_open_question.true_from_normalized_value(
                    ergo.logistic.sample_mixture(latest_prediction)
                )
                for _ in range(0, 5000)
            ]
        )

        assert np.mean(self.mock_samples) == pytest.approx(
            np.mean(prediction_samples), np.mean(prediction_samples) / 10
        )

    # smoke tests
    def test_get_community_prediction_linear(self):
        assert self.continuous_linear_closed_question.sample_community() > 0

    def test_get_community_prediction_log(self):
        assert self.continuous_log_open_question.sample_community() > 0


# Visual tests -- eyeball the results from these to see if they seem reasonable
# leave these commented out usually, just use them if they seem useful


# class TestVisualPandemic:
#     metaculus = ergo.Metaculus(uname, pwd, api_domain="pandemic")
#     sf_question = metaculus.get_question(3931, name="sf_question")
#     deaths_question = metaculus.get_question(3996)
#     show_performance_question = metaculus.get_question(4112)
#     show_performance_log_question = metaculus.get_question(4113)
#     mock_samples = np.array([ergo.logistic.sample_mixture(
#         tests.mocks.mock_true_params) for _ in range(0, 5000)])

#     def test_show_submission(self):
#         self.sf_question.show_submission(
#             self.mock_samples)

#     def test_show_submission_log(self):
#         self.deaths_question.show_submission(
#             self.mock_samples)

#     def test_show_performance(self):
#         # should have two humps, one on the left and one on the right
#         self.show_performance_question.show_performance()

#     def test_show_performance_log(self):
#         # should have a low, flat hump on the left and a skinny hump on the right
#         self.show_performance_log_question.show_performance()

#     def test_show_community_prediction(self):
#         self.sf_question.show_community_prediction()
