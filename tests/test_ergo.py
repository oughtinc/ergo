import ergo

test_uname = "oughttest"
test_pwd = "6vCo39Mz^rrb"
test_user_id = 112420


def test_version():
    assert ergo.__version__ == '0.2.0'


def test_login():
    metaculus = ergo.Metaculus(test_uname, test_pwd)
    assert metaculus.user_id == test_user_id
