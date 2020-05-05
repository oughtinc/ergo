import ergo


def test_nomem():
    """
    Without mem, different calls to foo() should differ sometimes
    """

    def foo():
        return ergo.lognormal_from_interval(1, 10)

    def model():
        x = foo()
        y = foo()
        return x == y

    samples = ergo.run(model, num_samples=1000)
    assert sum(samples["output"]) != 1000


def test_mem():
    """
    With mem, different calls to foo() should always have the same value
    """

    @ergo.mem
    def foo():
        return ergo.lognormal_from_interval(1, 10)

    def model():
        x = foo()
        y = foo()
        return x == y

    samples = ergo.run(model, num_samples=1000)
    assert sum(samples["output"]) == 1000


def test_mem_2():
    """
    Check that mem is cleared at the start of each run
    """

    @ergo.mem
    def model():
        return ergo.lognormal_from_interval(1, 10)

    samples = ergo.run(model, num_samples=1000)
    assert samples["output"].unique().size == 1000
