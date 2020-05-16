from ergo.distributions.logistic import Logistic, LogisticMixture

mock_mixture = LogisticMixture(
    components=[Logistic(loc=10000, scale=1000), Logistic(loc=100000, scale=10000)],
    probs=[0.8, 0.2],
)

mock_normalized_mixture = LogisticMixture(
    components=[
        Logistic(loc=0.15, scale=0.037034005),
        Logistic(loc=0.85, scale=0.032395907),
    ],
    probs=[0.6, 0.4],
)

mock_log_question_data = {
    "id": 0,
    "possibilities": {
        "type": "continuous",
        "scale": {"deriv_ratio": 10, "min": 1, "max": 10},
    },
    "title": "question_title",
}
