import jax.numpy as np
import numpy as onp
import pytest

from ergo import Logistic, LogisticMixture
from ergo.distributions import truncate
from ergo.distributions.conditions import HistogramCondition


def test_cdf():
    jax_dist = Logistic(loc=10, scale=1)
    original_dist = jax_dist.rv()
    for x in range(5, 15):
        assert jax_dist.cdf(x) == pytest.approx(original_dist.cdf(x), rel=0.1)


def test_fit_single_scipy():
    dist = Logistic.from_samples_scipy(onp.array([0.1, 0.2]))
    assert dist.loc == pytest.approx(0.15, abs=0.02)


def test_logpdf1(logistic_mixture):
    assert -18 < logistic_mixture.logpdf1(0.1) < -17


def test_logpdf(logistic_mixture):
    logistic_mixture.logpdf(np.array([0.1, 0.2]))


def test_fit_single_jax():
    dist = Logistic.from_samples(np.array([0.1, 0.2]))
    assert dist.loc == pytest.approx(0.15, abs=0.02)


def test_fit_single_compare():
    scipy_dist = Logistic.from_samples_scipy(onp.array([0.1, 0.2]))
    jax_dist = Logistic.from_samples(np.array([0.1, 0.2]))
    assert scipy_dist.loc == pytest.approx(float(jax_dist.loc), abs=0.1)
    assert scipy_dist.scale == pytest.approx(float(jax_dist.scale), abs=0.1)


def test_fit_mixture_small():
    mixture = LogisticMixture.from_samples(
        np.array([0.1, 0.2, 0.8, 0.9]), num_components=2
    )
    for prob in mixture.probs:
        assert prob == pytest.approx(0.5, 0.1)
    locs = sorted([component.loc for component in mixture.components])
    assert locs[0] == pytest.approx(0.15, abs=0.1)
    assert locs[1] == pytest.approx(0.85, abs=0.1)


def test_fit_mixture_large():
    data1 = onp.random.logistic(loc=0.7, scale=0.1, size=1000)
    data2 = onp.random.logistic(loc=0.4, scale=0.2, size=1000)
    data = onp.concatenate([data1, data2])
    mixture = LogisticMixture.from_samples(data, num_components=2)
    locs = sorted([component.loc for component in mixture.components])
    scales = sorted([component.scale for component in mixture.components])
    assert locs[0] == pytest.approx(0.4, abs=0.2)
    assert locs[1] == pytest.approx(0.7, abs=0.2)
    assert scales[0] == pytest.approx(0.1, abs=0.2)
    assert scales[1] == pytest.approx(0.2, abs=0.2)


def test_mixture_cdf():
    # Make a mixture with known properties. The median should be 15 for this mixture.
    logistic_params = np.array([[10, 3.658268, 0.5], [20, 3.658268, 0.5]])
    mixture = LogisticMixture.from_params(logistic_params)
    cdf50 = mixture.cdf(15)
    assert cdf50 == pytest.approx(0.5, rel=1e-3)


def test_mixture_ppf():
    # Make a mixtures with known properties. The median should be 10 for this mixture.
    logistic_params = np.array([[15, 2.3658268, 0.5], [5, 2.3658268, 0.5]])
    mixture = LogisticMixture.from_params(logistic_params)
    ppf5 = mixture.ppf(0.5)
    assert ppf5 == pytest.approx(10, rel=1e-3)


def test_mixture_ppf_adversarial():
    # Make a mixture with one very improbable distribution and one dominant
    mixture = LogisticMixture([Logistic(10, 3), Logistic(5, 5)], [1.8629593e-29, 1.0])
    assert mixture.ppf(0.5) == pytest.approx(5.0, rel=1e-3)
    assert mixture.ppf(0.01) == pytest.approx(-17.9755, rel=1e-3)
    assert mixture.ppf(0.001) == pytest.approx(-29.5337, rel=1e-3)
    assert mixture.ppf(0.99) == pytest.approx(27.9755, rel=1e-3)
    assert mixture.ppf(0.999) == pytest.approx(39.5337, rel=1e-3)

    # Make a mixture with hugely overlapping distributions
    mixture = LogisticMixture(
        [
            Logistic(4000000.035555004, 200000.02),
            Logistic(4000000.0329152746, 200000.0),
        ],
        [0.5, 0.5],
    )
    assert mixture.ppf(0.5) == pytest.approx(4000000.0342351394, rel=1e-3)
    assert mixture.ppf(0.01) == pytest.approx(3080976.018257023, rel=1e-3)
    assert mixture.ppf(0.001) == pytest.approx(2618649.009437881, rel=1e-3)
    assert mixture.ppf(0.99) == pytest.approx(4919024.050213255, rel=1e-3)
    assert mixture.ppf(0.999) == pytest.approx(5381351.059032397, rel=1e-3)


def ppf_cdf_round_trip():
    mixture = LogisticMixture.from_samples(
        np.array([0.5, 0.4, 0.8, 0.8, 0.9, 0.95, 0.15, 0.1]), num_components=3
    )
    x = 0.65
    prob = mixture.cdf(x)
    assert mixture.ppf(prob) == pytest.approx(x, rel=1e-3)


@pytest.mark.xfail(reason="Fitting to samples doesn't reliably work yet #219")
def test_fit_samples(logistic_mixture):
    data = np.array([logistic_mixture.sample() for _ in range(0, 1000)])
    fitted_mixture = LogisticMixture.from_samples(data, num_components=2)
    true_locs = sorted([c.loc for c in logistic_mixture.components])
    true_scales = sorted([c.scale for c in logistic_mixture.components])
    fitted_locs = sorted([c.loc for c in fitted_mixture.components])
    fitted_scales = sorted([c.scale for c in fitted_mixture.components])
    for (true_loc, fitted_loc) in zip(true_locs, fitted_locs):
        assert fitted_loc == pytest.approx(float(true_loc), rel=0.2)
    for (true_scale, fitted_scale) in zip(true_scales, fitted_scales):
        assert fitted_scale == pytest.approx(float(true_scale), rel=0.2)


def test_logistic_mixture_normalization():
    mixture = LogisticMixture([Logistic(-40, 1), Logistic(50, 10)], [0.5, 0.5])

    for scale_min, scale_max in [(0, 10), (10, 100), (-10, 10), (-100, -10)]:
        assert (
            mixture.normalize(scale_min, scale_max).denormalize(scale_min, scale_max)
            == mixture
        )

    normalized = mixture.normalize(-50, 50)
    assert normalized == LogisticMixture(
        [Logistic(0.1, 0.01), Logistic(1, 0.1)], [0.5, 0.5]
    )


def make_histogram(rv, xmin, xmax, num_bins):
    """
    Make a histogram based on a random variable and a range

    :param rv: random variable to sample the histogram from
    :param xmin: range minimum
    :param xmax: range maximum
    :param num_bins: number of bins in the histogram
    :return: histogram bins
    """
    xs = np.linspace(xmin, xmax, num_bins)
    ys = [float(rv.pdf1(x)) for x in xs]
    return [{"x": x, "density": y} for (x, y) in zip(xs, ys)]


def test_fit_hist_with_p_on_edge(verbose=True):
    scale_min = 0
    scale_max = 10
    num_bins = 100
    xs = onp.linspace(scale_min, scale_max, num_bins)
    densities = ([1] * 10) + ([0] * 90)
    test_hist_condition = HistogramCondition(xs, densities)

    mixture = truncate(LogisticMixture).from_conditions(
        conditions=[test_hist_condition],
        scale_min=scale_min,
        scale_max=scale_max,
        num_components=3,
        verbose=verbose,
        init_tries=100,
        opt_tries=2,
    )

    mixture_hist_bins = make_histogram(mixture, scale_min, scale_max, num_bins)
    # there's some sort of off-by-one-type error with
    # the xs for the mixture hist and the test_hist,
    # ignoring for now

    density_diff = [
        abs(bin["density"] - densities[idx])
        for idx, bin in enumerate(mixture_hist_bins)
    ]

    print(density_diff)

    print(max(density_diff))

    loss = sum([bin_fit ** 2 for bin_fit in density_diff]) / len(density_diff)

    assert loss < 0.1


def test_speed():
    from ergo.distributions.compile import compile
    import cProfile
    import time

    compiled = compile()

    start = time.time()
    test_fit_hist_with_p_on_edge(verbose=False)
    print(f"Runtime: {time.time() - start}s")

    cProfile.runctx(
        "test_fit_hist_with_p_on_edge(compiled)",
        {
            "test_fit_hist_with_p_on_edge": test_fit_hist_with_p_on_edge,
            "compiled": compiled,
        },
        {},
        "test_truncated_logistic.prof",
    )
