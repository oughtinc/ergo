# https: // www.metaculus.com/questions/4111/for-testing-log-question-open-both-ends-1000-to-2000/

from scipy import stats
true_loc = 1300
true_25th = 1000
true_75th = 1700

prediction_request = {"prediction": {"kind": "multi", "d": [
    {"kind": "logistic", "x0": 0.376, "s": 0.3727, "w": 1, "low": 0.248, "high": 0.863}]}, "void": False}

submission = prediction_request["prediction"]["d"][0]

submission_loc = submission["x0"]
submission_scale = submission["s"]

abridged_question_json = {
    "url": "https://www.metaculus.com/api2/questions/4111/",
    "page_url": "/questions/4111/for-testing-log-question-open-both-ends-1000-to-2000/",
    "id": 4111,
    "author": 112420,
    "title": "(for testing) log question open both ends 1000 to 2000",
    "status": "V",
    "created_time": "2020-04-12T02:46:55.277027Z",
    "publish_time": "2020-04-12T02:46:55.273661Z",
    "close_time": "2099-01-01T08:00:00Z",
    "resolve_time": "2099-01-01T08:00:00Z",
    "possibilities": {
        "low": "tail",
        "high": "tail",
        "type": "continuous",
        "scale": {
            "max": 2000,
            "min": 1000,
            "deriv_ratio": 2
        },
        "format": "num"
    },
    "last_activity_time": "2020-04-12T02:49:01.996856Z",
    "activity": 19.959300466782963,
    "comment_count": 0,
    "votes": 0,
    "prediction_timeseries": [
        {
            "t": 1586659742.0495358,
            "distribution": {
                "s": 0.3727,
                "x0": 0.37602,
                "low": 0.248,
                "num": 1,
                "high": 0.863,
                "kind": "logistic"
            },
            "num_predictions": 1,
            "community_prediction": {
                "q1": 0.00362,
                "q2": 0.37608,
                "q3": 0.74858,
                "low": 0.248,
                "high": 0.863
            }
        }
    ],
    "user_vote": 0,
    "user_community_vis": 0,
    "my_predictions": {
        "id": 122806,
        "predictions": [
            {
                "d": [
                    {
                        "s": 0.3727,
                        "w": 1,
                        "x0": 0.376,
                        "low": 0.248,
                        "high": 0.863,
                        "kind": "logistic"
                    }
                ],
                "t": 1586659741.9905374,
                "kind": "multi"
            }
        ],
        "user": 112420,
        "question": 4111,
    },
    "author_name": "oughttest",
    "anon_prediction_count": 0,
    "last_read": "2020-04-12T02:46:55.469870Z"
}

scale = abridged_question_json["possibilities"]["scale"]


def get_true_loc(submission_loc, scale):
    expd = scale["deriv_ratio"] ** submission_loc

    width = scale["max"] - scale["min"]
    scaled = expd * width

    return scaled


# print(get_true_loc(submission_loc, scale))

prediction_request_loc_1900 = {"prediction": {"kind": "multi", "d": [
    {"kind": "logistic", "x0": 0.9278, "s": 0.3727, "w": 1, "low": 0.061, "high": 0.553}]}, "void": False}

submission_median_1900 = prediction_request_loc_1900["prediction"]["d"][0]

# print(get_true_loc(submission_median_1900["x0"], scale))


def get_true_scale(submission_scale, scale):
    expd = scale["deriv_ratio"] ** submission_scale

    width = scale["max"] - scale["min"]
    scaled = expd * width

    return scaled


submission_dist = stats.logistic(submission_loc, submission_scale)


def get_true_value_from_sub_dist_value(sub_dist_value, scale):
    expd = scale["deriv_ratio"] ** sub_dist_value

    width = scale["max"] - scale["min"]
    scaled = expd * width

    return scaled


print(submission_dist.ppf(0.5))

print(get_true_value_from_sub_dist_value(submission_dist.ppf(0.5), scale))

print(get_true_value_from_sub_dist_value(submission_dist.ppf(0.25), scale))

print(get_true_value_from_sub_dist_value(submission_dist.ppf(0.75), scale))
