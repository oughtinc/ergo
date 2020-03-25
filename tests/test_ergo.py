import ergo

test_uname = "oughttest"
test_pwd = "6vCo39Mz^rrb"
test_user_id = 112420


def test_version():
    assert ergo.__version__ == '0.2.0'


def test_login():
    metaculus = ergo.Metaculus(test_uname, test_pwd)
    assert metaculus.user_id == test_user_id


def test_submission():
    metaculus = ergo.Metaculus(test_uname, test_pwd)

    # need to replace this with a different question once this one resolves on Mar 27
    euro_question = metaculus.get_question(3706)
    r = euro_question.submit(0.534894790856232, 0.02)
    assert r.status_code == 202
