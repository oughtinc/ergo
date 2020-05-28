import jax.numpy as np

from ergo.distributions.conditions import (
    CrossEntropyCondition,
    IntervalCondition,
    MaxEntropyCondition,
    SmoothnessCondition,
)
from ergo.distributions.histogram import HistogramDist


def single_interval_condition_cases():
    for lower_bound in [None, 0.3]:
        for upper_bound in [None, 0.9]:
            yield IntervalCondition(p=0.5, min=lower_bound, max=upper_bound)


def compile_histogram_loss_functions(num_bins: int = 201):
    print("Compiling loss functions...")
    dist = HistogramDist(np.array([-float(num_bins)] * num_bins))
    conditions = list(single_interval_condition_cases()) + [
        CrossEntropyCondition(dist),
        MaxEntropyCondition(),
        SmoothnessCondition(weight=10.0),
    ]
    for condition in conditions:
        dist = HistogramDist.from_conditions(
            [condition], scale_min=0, scale_max=1, num_bins=num_bins
        )
        condition.describe_fit(dist)
    print("Compilation done!")


def compile():
    compile_histogram_loss_functions()
