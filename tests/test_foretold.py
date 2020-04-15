import ergo
import pytest
import numpy as np


class TestForetold:
    def test_foretold_sampling(self):
        dist = ergo.ForetoldDistribution("cf86da3f-c257-4787-b526-3ef3cb670cb4")
        num_samples = 2000
        samples = ergo.run(
            lambda: ergo.tag(dist.sample(), "sample"), num_samples=num_samples
        )
        assert np.count_nonzero(samples > 100) == pytest.approx(num_samples / 2, 0.1)

