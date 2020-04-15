import pandas as pd
import numpy as np
import ergo

mock_true_params = ergo.logistic.LogisticMixtureParams(
    components=[ergo.logistic.LogisticParams(loc=10000, scale=1000),
                ergo.logistic.LogisticParams(loc=100000, scale=10000)],
    probs=[0.8, 0.2]
)

mock_normalized_params = ergo.logistic.LogisticMixtureParams(
    components=[ergo.logistic.LogisticParams(loc=0.15, scale=0.037034005),
                ergo.logistic.LogisticParams(loc=0.85, scale=0.032395907)],
    probs=[0.6, 0.4]
)

mock_log_question_data = {
    "id": 0,
    "possibilities": {
        "type": "continuous",
        "scale": {
            "deriv_ratio": 10,
            "min": 1,
            "max": 10
        }
    }
}
