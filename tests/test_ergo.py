import ergo
import pytest  # type: ignore
import requests
import pendulum
import pprint

pp = pprint.PrettyPrinter(indent=4)

test_uname = "oughttest"
test_pwd = "6vCo39Mz^rrb"
test_user_id = 112420

mock_sample = [1, 2, 3]


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

    def test_submission_continuous_linear_open(self):
        r = self.continuous_linear_open_question.submit(
            0.534894790856232, 0.02)
        assert r.status_code == 202

    def test_submission_continuous_linear_closed(self):
        r = self.continuous_linear_closed_question.submit(
            0.534894790856232, 0.02)
        assert r.status_code == 202

    def test_submission_continuous_log_open(self):
        r = self.continuous_log_open_question.submit(
            0.534894790856232, 0.02)
        assert r.status_code == 202

    def test_submission_for_closed_question_fails(self):
        with pytest.raises(requests.exceptions.HTTPError):
            r = self.closed_question.submit(
                0.534894790856232, 0.02)

    def test_score_binary(self):
        # smoke test
        self.binary_question.get_scored_predictions()

    def test_score_continuous(self):
        # smoke test
        self.continuous_linear_open_question.get_scored_predictions()

    def test_show_prediction_results(self):
        # smoke test
        self.metaculus.get_prediction_results()

    def test_get_questions(self):
        questions = self.metaculus.get_questions()
        assert len(questions) >= 20

    def test_get_questions_pages(self):
        two_pages = self.metaculus.get_questions(pages=2)
        assert len(two_pages) >= 40

    def test_get_questions_end_of_pages(self):
        all_pages = self.metaculus.get_questions(
            player_status="predicted", pages=9999)
        # basically just a smoke test to make sure it returns some results and doesn't just error
        assert len(all_pages) > 1

    def test_get_questions_player_status(self):
        i_predicted = self.metaculus.get_questions(player_status="predicted")
        for q in i_predicted:
            assert q["my_predictions"] is not None

        not_predicted = self.metaculus.get_questions(
            player_status="not-predicted")
        for q in not_predicted:
            assert q["my_predictions"] is None

    def test_get_questions_question_status(self):
        open = self.metaculus.get_questions(question_status="open")
        for q in open:
            assert pendulum.parse(open[0]["close_time"]) > pendulum.now()

        closed = self.metaculus.get_questions(question_status="closed")
        for q in closed:
            assert pendulum.parse(closed[0]["close_time"]) < pendulum.now()


class TestPPL:
    def test_sampling(self):
        @ergo.model
        def model():
            x = ergo.lognormal_from_interval(1, 10, name="x")
            y = ergo.beta_from_hits(2, 10, name="y")
            z = x * y  
            ergo.tag(z, "z")
        samples = ergo.run(model, num_samples=10000)
        stats = samples.describe()
        assert 3.5 < stats["x"]["mean"] < 4.5
        assert 0.1 < stats["y"]["mean"] < 0.3
        assert 0.6 < stats["z"]["mean"] < 1.0


class TestData:
    def test_confirmed_infections(self):
        confirmed = ergo.data.covid19.ConfirmedInfections()
        assert confirmed.get("Iran", "3/25/20") == 27017
