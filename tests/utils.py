import numpyro


def random_seed(fn):
    def wrapped(*args, **kwargs):
        with numpyro.handlers.seed(rng_seed=0):
            return fn(*args, **kwargs)

    return wrapped
