from ergo.scale import LogScale, Scale


def test_serialization():
    assert hash(Scale(0, 100)) == hash(Scale(0, 100))
    assert hash(Scale(0, 100)) != hash(Scale(100, 200))
    assert hash(LogScale(0, 100, 1)) != hash(Scale(0, 100))
    assert hash(LogScale(0, 100, 10)) != hash(LogScale(0, 100, 10))
