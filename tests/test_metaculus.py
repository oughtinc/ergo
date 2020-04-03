import ergo
import pytest
import requests
import pendulum
import pprint
import tests.mocks

pp = pprint.PrettyPrinter(indent=4)

test_uname = "oughttest"
test_pwd = "6vCo39Mz^rrb"
test_user_id = 112420


class TestMetaculus:
    metaculus = ergo.Metaculus(test_uname, test_pwd)
    continuous_linear_closed_question = metaculus.get_question(3963)
    continuous_linear_open_question = metaculus.get_question(3962)
    continuous_log_open_question = metaculus.get_question(3961)
    closed_question = metaculus.get_question(3965)
    binary_question = metaculus.get_question(3966)

    def test_login(self):
        assert self.metaculus.user_id == test_user_id

    def test_get_question(self):
        # make sure we're getting the user-specific data
        assert "my_predictions" in self.continuous_linear_open_question.data

    def test_submit_continuous_linear_open(self):
        submission = self.continuous_linear_open_question.get_submission(
            0.534894790856232, 0.02)
        r = self.continuous_linear_open_question.submit(submission)
        assert r.status_code == 202

    def test_submit_continuous_linear_closed(self):
        submission = self.continuous_linear_closed_question.get_submission(
            0.534894790856232, 0.02)
        r = self.continuous_linear_closed_question.submit(submission)
        assert r.status_code == 202

    def test_submit_continuous_log_open(self):
        submission = self.continuous_log_open_question.get_submission(
            0.534894790856232, 0.02)
        r = self.continuous_log_open_question.submit(submission)
        assert r.status_code == 202

    def test_submit_binary(self):
        r = self.binary_question.submit(0.95)
        assert r.status_code == 202

    def test_submit_closed_question_fails(self):
        with pytest.raises(requests.exceptions.HTTPError):
            submission = self.closed_question.get_submission(
                0.534894790856232, 0.02)
            r = self.closed_question.submit(submission)
            print(r)

    def test_score_binary(self):
        # smoke test
        self.binary_question.get_scored_predictions()

    def test_score_continuous(self):
        # smoke test
        self.continuous_linear_open_question.get_scored_predictions()

    def test_get_prediction_results(self):
        # smoke test
        self.metaculus.get_prediction_results()

    def test_get_questions_json(self):
        questions = self.metaculus.get_questions_json()
        assert len(questions) >= 20

    def test_get_questions_json_pages(self):
        two_pages = self.metaculus.get_questions_json(pages=2)
        assert len(two_pages) >= 40

    def test_get_questions_player_status(self):
        qs_i_predicted = self.metaculus.make_questions_df(self.metaculus.get_questions_json(
            player_status="predicted"))
        assert qs_i_predicted["i_predicted"].all()

        not_predicted = self.metaculus.make_questions_df(self.metaculus.get_questions_json(
            player_status="not-predicted"))
        assert (not_predicted["i_predicted"] == False).all()

    def test_get_questions_question_status(self):
        open = self.metaculus.make_questions_df(self.metaculus.get_questions_json(
            question_status="open"))
        assert(open["close_time"] > pendulum.now()).all()

        closed = self.metaculus.make_questions_df(
            self.metaculus.get_questions_json(question_status="closed"))
        assert(closed["close_time"] < pendulum.now()).all()

# Visual tests -- eyeball the results from these to see if they seem reasonable
# leave these commented out usually, just use them if they seem useful


class TestPandemic:
    metaculus = ergo.Metaculus(test_uname, test_pwd, api_domain="pandemic")
    sf_question = metaculus.get_question(3931)

    # def test_show_submission(self):
    #     self.sf_question.show_submission(
    #         tests.mocks.samples)

    # def test_show_performance(self):
    #     self.sf_question.show_performance()
