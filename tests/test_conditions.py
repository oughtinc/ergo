import pytest

from ergo import LogisticMixture
from ergo.distributions.conditions import PercentileCondition


def test_percentile_condition():
    for value in [0.01, 0.1, 1, 3]:
        conditions = [PercentileCondition(percentile=0.5, value=value)]
        dist = LogisticMixture.from_conditions(
            conditions, num_components=1, verbose=True
        )
        loc = dist.components[0].loc
        assert loc == pytest.approx(value, rel=0.1), loc
