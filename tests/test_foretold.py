import numpy as np
import pytest

import ergo


class TestForetold:
    def test_foretold_sampling(self):
        foretold = ergo.Foretold()
        # https://www.foretold.io/c/f45577e4-f1b0-4bba-8cf6-63944e63d70c/m/cf86da3f-c257-4787-b526-3ef3cb670cb4
        # Distribution is mm(10 to 20, 200 to 210), a mixture model with most mass split between
        # 10 - 20 and 200 - 210.
        dist = foretold.get_question("cf86da3f-c257-4787-b526-3ef3cb670cb4")
        assert dist.quantile(0.25) < 100
        assert dist.quantile(0.75) > 100

        num_samples = 20000
        samples = ergo.run(
            lambda: ergo.tag(dist.sample_community(), "sample"), num_samples=num_samples
        )
        # Probability mass is split evenly between both modes of the distribution, so approximately half of the
        # samples should be lower than 100
        assert np.count_nonzero(samples > 100) == pytest.approx(num_samples / 2, 0.1)
