import ergo


def test_nomem():
    def foo():
        return ergo.lognormal_from_interval(1, 10)

    def model():
        x = foo()
        y = foo()
        return x == y

    samples = ergo.run(model, num_samples=1000)
    assert sum(samples["output"]) != 1000


def test_mem():
    @ergo.mem
    def foo():
        return ergo.lognormal_from_interval(1, 10)

    def model():
        x = foo()
        y = foo()
        return x == y

    samples = ergo.run(model, num_samples=1000)
    assert sum(samples["output"]) == 1000
