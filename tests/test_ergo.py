import ergo
import pytest
import requests

test_uname = "oughttest"
test_pwd = "6vCo39Mz^rrb"
test_user_id = 112420

mock_sample = [1, 2, 3]


def test_version():
    assert ergo.__version__ == '0.2.0'


class TestMetaculus:
    metaculus = ergo.Metaculus(test_uname, test_pwd)
    euro_question = metaculus.get_question(3706)
    uk_question = metaculus.get_question(3761)

    def test_login(self):
        assert self.metaculus.user_id == test_user_id

    def test_submission(self):
        euro_response = self.euro_question.submit(
            0.534894790856232, 0.02)
        assert euro_response.status_code == 202

    def test_submission_for_closed_question_fails(self):
        with pytest.raises(requests.exceptions.HTTPError):
            uk_response = self.uk_question.submit(
                0.534894790856232, 0.02)
            print(uk_response.status_code, uk_response.json())

    def test_show_submission(self):
        self.euro_question.show_submission(mock_sample)
