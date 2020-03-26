import ergo

test_uname = "oughttest"
test_pwd = "6vCo39Mz^rrb"
test_user_id = 112420

mock_sample = [1, 2, 3]


def test_version():
    assert ergo.__version__ == '0.2.0'


class TestMetaculus:
    metaculus = ergo.Metaculus(test_uname, test_pwd)
    euro_question = metaculus.get_question(3706)

    def test_login(self):
        assert self.metaculus.user_id == test_user_id

    def test_submission(self):
        r = self.euro_question.submit(0.534894790856232, 0.02)
        assert r.status_code == 202

    def test_show_submission(self):
        self.euro_question.show_submission(mock_sample)
