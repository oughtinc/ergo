import ergo


def test_rejection():
    def model():
        x = ergo.flip()
        y = ergo.flip()
        ergo.condition(x or y)
        return x == y

    samples = ergo.run(model, num_samples=1000)
    assert 266 < sum(samples["output"]) < 466
