from ergo.scale import LogScale, Scale, scale_factory


def test_serialization():
    assert hash(Scale(0, 100)) == hash(Scale(0, 100))
    assert hash(Scale(0, 100)) != hash(Scale(100, 200))
    assert hash(LogScale(0, 100, 1)) != hash(Scale(0, 100))
    assert hash(LogScale(0, 100, 10)) != hash(LogScale(0, 100, 10))


def test_export_import():
    log_scale = LogScale(low=-1, high=1, log_base=2)
    log_scale_export = log_scale.export()
    assert log_scale_export["width"] == 2
    assert log_scale_export["class"] == "LogScale"

    assert (scale_factory(log_scale.export())) == log_scale

    linear_scale = Scale(low=1, high=10000)
    assert (scale_factory(linear_scale.export())) == linear_scale
