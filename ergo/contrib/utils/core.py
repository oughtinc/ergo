from datetime import date, timedelta
import functools

import numpy as np

import ergo

# TODO consider turning this into a Class or factoring into Ergo proper

# Rejection sampling


def rejection_sample(fn, condition):
    """
    Sample from fn until we get a value that satisfies
    condition, then return it.
    """
    while True:
        value = fn()
        if condition(value):
            return value


memoized_functions = []


def mem(func):
    func = functools.lru_cache(None)(func)
    memoized_functions.append(func)
    return func


def clear_mem():
    for func in memoized_functions:
        func.cache_clear()


# Associate models with questions

# We'll add a sampler here for each question we predict on.
# Each sampler is a function that returns a single sample
# from our model predicting on that question.
samplers = {}


# TODO probably curry this with the notbooks metaculus instance so we don't need to pass
# it in on every question
def question(
    metaculus,
    question_id,
    community_weight=0,
    community_fn=None,
    start_date=date.today(),
):
    q = metaculus.get_question(question_id)

    def decorator(func):
        tag = func.__name__

        @functools.wraps(func)
        @mem
        def sampler():
            if ergo.flip(community_weight):
                if community_fn:
                    value = community_fn()
                else:
                    value = q.sample_community()
            else:
                value = func()
            if isinstance(value, date):
                # FIXME: Ergo needs to handle dates
                ergo.tag(int((value - start_date).days), tag)
            else:
                ergo.tag(value, tag)
            return value

        sampler.question = q
        samplers[q.id] = sampler
        return sampler

    return decorator


def summarize_question_samples(samples):
    sampler_tags = [sampler.__name__ for sampler in samplers.values()]
    tags_to_show = [tag for tag in sampler_tags if tag in samples.columns]
    samples_to_show = samples[tags_to_show]
    summary = samples_to_show.describe().transpose().round(2)
    display(summary)  # noqa: F821   #TODO see if we need this display command


def plot_question(sampler, num_samples=200, bw=None, start_date=date.today()):
    def model():
        clear_mem()
        sampler()

    samples = ergo.run(model, num_samples=num_samples)

    summarize_question_samples(samples)

    q = sampler.question

    q_samples = samples[sampler.__name__]

    if (
        q.id == 4128
    ):  # Date question: Need to convert back to date from days (https://github.com/oughtinc/ergo/issues/144)
        q_samples = np.array([start_date + timedelta(s) for s in q_samples])

    if bw is not None:
        q.show_prediction(
            samples=q_samples, show_community=True, percent_kept=0.9, bw=bw
        )
    else:
        q.show_prediction(samples=q_samples, show_community=True, percent_kept=0.9)


def sample_from_ensemble(models, params, weights=None, fallback=False, default=None):
    """Sample models in proportion to weights and execute with
    model_params. If fallback is true then call different model from
    ensemble if the selected model throws an error. If Default is not
    None then return default if all models fail

    """

    model = ergo.random_choice(models, weights)
    try:
        result = model(**params)
        if np.isnan(result):
            raise KeyError
        return result
    except KeyError:
        if fallback and len(models) > 1:
            i = models.index(model)
            del models[i]
            del weights[i]
            return sample_from_ensemble(models, params, weights, fallback, default)
        return default
