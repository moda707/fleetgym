"""Samplers: goodness of fit at fixed seeds (alpha = 1e-3), means and quantiles against scipy."""

import math

import numpy as np
import pytest
from scipy import stats

from fleetgym.spec.registries import DISTRIBUTIONS
from fleetgym.uncertainty.distributions import (
    DISTRIBUTION_CLASSES,
    build_distribution,
)

N = 20_000
ALPHA = 1e-3

CONTINUOUS = [
    ("uniform", {"low": 2.0, "high": 5.0}, stats.uniform(loc=2.0, scale=3.0)),
    ("normal", {"mean": 1.0, "std": 2.0}, stats.norm(loc=1.0, scale=2.0)),
    (
        "truncated_normal",
        {"mean": 1.0, "std": 2.0, "low": 0.0, "high": 4.0},
        stats.truncnorm(-0.5, 1.5, loc=1.0, scale=2.0),
    ),
    (
        "truncated_normal",
        {"mean": 10.0, "std": 3.0, "low": 8.0, "high": None},
        stats.truncnorm(-2 / 3, np.inf, loc=10.0, scale=3.0),
    ),
    ("lognormal", {"mu": 0.5, "sigma": 0.4}, stats.lognorm(s=0.4, scale=math.exp(0.5))),
    ("gamma", {"shape": 2.5, "mean": 5.0}, stats.gamma(a=2.5, scale=2.0)),
    ("weibull", {"shape": 1.7, "scale": 3.0}, stats.weibull_min(c=1.7, scale=3.0)),
    ("exponential", {"mean": 4.0}, stats.expon(scale=4.0)),
]


def rng(seed: int = 12345) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.mark.parametrize(("name", "params", "reference"), CONTINUOUS)
def test_continuous_samplers_fit_their_distribution(name, params, reference):
    dist = build_distribution(name, params)
    draws = dist.sample(rng(), N)
    assert draws.shape == (N,)
    assert stats.kstest(draws, reference.cdf).pvalue > ALPHA
    np.testing.assert_allclose(dist.mean(3), reference.mean(), rtol=1e-12)
    for q in (0.05, 0.5, 0.95):
        np.testing.assert_allclose(dist.quantile(q, 2), reference.ppf(q), rtol=1e-9)
    assert dist.support(3) is None


def chi_square_pvalue(draws: np.ndarray, values: np.ndarray, probs: np.ndarray) -> float:
    observed = np.array([np.sum(draws == v) for v in values])
    assert observed.sum() == draws.size  # no draw outside the support
    keep = probs * draws.size >= 5
    expected = probs[keep] * draws.size
    observed_kept = observed[keep]
    expected = expected * observed_kept.sum() / expected.sum()
    return float(stats.chisquare(observed_kept, expected).pvalue)


DISCRETE = [
    ("bernoulli", {"p": 0.3}, np.array([0.0, 1.0]), np.array([0.7, 0.3])),
    (
        "categorical",
        {"values": (1.0, 4.0, 9.0), "probs": (0.2, 0.5, 0.3)},
        np.array([1.0, 4.0, 9.0]),
        np.array([0.2, 0.5, 0.3]),
    ),
    (
        "empirical",
        {"values": (3.0, 3.0, 7.0, 10.0), "weights": None},
        np.array([3.0, 7.0, 10.0]),
        np.array([0.5, 0.25, 0.25]),
    ),
    (
        "empirical",
        {"values": (1.0, 2.0), "weights": (1.0, 3.0)},
        np.array([1.0, 2.0]),
        np.array([0.25, 0.75]),
    ),
]


@pytest.mark.parametrize(("name", "params", "values", "probs"), DISCRETE)
def test_discrete_samplers_fit_their_distribution(name, params, values, probs):
    dist = build_distribution(name, params)
    draws = dist.sample(rng(), N)
    assert chi_square_pvalue(draws, values, probs) > ALPHA
    np.testing.assert_allclose(dist.mean(2), np.dot(values, probs))
    cdf = np.cumsum(probs)
    for q in (0.1, 0.5, 0.9, 1.0):
        expected = values[int(np.searchsorted(cdf, q - 1e-12))]
        assert dist.quantile(q, 1)[0] == expected
    support = dist.support(2)
    assert support is not None and len(support) == 2
    for vals, ps in support:
        merged = dict.fromkeys(values, 0.0)
        for v, p in zip(vals, ps, strict=True):
            merged[v] += p
        np.testing.assert_allclose([merged[v] for v in values], probs)


def test_poisson_fits_and_supports_vector_means():
    dist = build_distribution("poisson", {"mean": 3.5})
    draws = dist.sample(rng(), N)
    values = np.arange(0, 30, dtype=float)
    probs = stats.poisson.pmf(values, 3.5)
    probs = probs / probs.sum()
    assert chi_square_pvalue(draws, values, probs) > ALPHA
    assert dist.quantile(0.5, 1)[0] == stats.poisson.ppf(0.5, 3.5)
    assert dist.support(2) is None
    per_node = build_distribution("poisson", {"mean": np.array([0.0, 2.0, 50.0])})
    many = np.array([per_node.sample(rng(s), 3) for s in range(2000)])
    assert np.all(many[:, 0] == 0)
    np.testing.assert_allclose(many.mean(axis=0), [0.0, 2.0, 50.0], rtol=0.05)
    zero = build_distribution("poisson", {"mean": np.zeros(2)})
    assert zero.support(2) == [(np.array([0.0]), np.array([1.0]))] * 2


def test_fixed_consumes_no_randomness():
    gen = rng()
    state = gen.bit_generator.state
    dist = build_distribution("fixed", {"value": np.array([0.0, 4.0])})
    np.testing.assert_array_equal(dist.sample(gen, 2), [0.0, 4.0])
    assert gen.bit_generator.state == state
    assert dist.quantile(0.3, 2).tolist() == [0.0, 4.0]
    support = dist.support(2)
    assert support is not None
    assert [s[0].tolist() for s in support] == [[0.0], [4.0]]


def test_kde_samples_near_the_data():
    data = np.array([1.0, 2.0, 2.5, 3.0, 10.0])
    dist = build_distribution("kde", {"data": data, "bandwidth": 0.3})
    draws = dist.sample(rng(), N)
    reference = stats.gaussian_kde(data, bw_method=0.3)
    grid = np.linspace(-5, 20, 400)
    cdf = np.array([reference.integrate_box_1d(-np.inf, x) for x in grid])
    empirical = np.searchsorted(np.sort(draws), grid) / N
    assert np.max(np.abs(cdf - empirical)) < 0.02
    assert dist.mean(1)[0] == pytest.approx(data.mean())
    with pytest.raises(NotImplementedError):
        dist.quantile(0.5, 1)
    with pytest.raises(ValueError, match="one-dimensional"):
        build_distribution("kde", {"data": np.ones((3, 2))}).sample(rng(), 1)


def test_same_seed_same_draws_and_parameter_shapes():
    dist = build_distribution("gamma", {"shape": 2.0, "mean": np.array([1.0, 2.0, 3.0])})
    np.testing.assert_array_equal(dist.sample(rng(5), 3), dist.sample(rng(5), 3))
    with pytest.raises(ValueError, match="shape"):
        dist.sample(rng(), 4)
    with pytest.raises(ValueError, match="q must be"):
        dist.quantile(1.5, 3)


def test_every_spec_distribution_has_a_sampler():
    assert set(DISTRIBUTIONS.names()) <= set(DISTRIBUTION_CLASSES.names())
