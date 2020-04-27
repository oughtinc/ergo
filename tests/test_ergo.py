import pytest

import ergo


class TestPPL:
    @pytest.mark.xfail(
        reason="Either there is a bug or the lognormal parameterization has changed between Pyro and Numpyro"
    )
    def test_sampling(self):
        def model():
            x = ergo.lognormal_from_interval(1, 10, name="x")
            y = ergo.beta_from_hits(2, 10, name="y")
            z = x * y
            ergo.tag(z, "z")

        samples = ergo.run(model, num_samples=2000)
        stats = samples.describe()
        assert 3.5 < stats["x"]["mean"] < 4.5
        assert 0.1 < stats["y"]["mean"] < 0.3
        assert 0.6 < stats["z"]["mean"] < 1.0


class TestData:
    def test_confirmed_infections(self):
        confirmed = ergo.data.covid19.ConfirmedInfections()
        assert confirmed.get("Iran", "3/25/20") == 27017
