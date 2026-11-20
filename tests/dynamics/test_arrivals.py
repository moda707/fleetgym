"""Arrival processes: KS and chi-square tests at fixed seeds (alpha = 1e-3)."""

import itertools

import numpy as np
import pytest
from scipy import stats

from fleetgym.uncertainty.arrivals import (
    expected_count,
    piecewise_rate,
    poisson_arrivals,
    thinned_arrivals,
)
from fleetgym.uncertainty.dynamism import degree_of_dynamism, effective_degree_of_dynamism

ALPHA = 1e-3


def poisson_counts_pvalue(observed: np.ndarray, expected: np.ndarray) -> float:
    """Chi-square test of independent Poisson counts against their means (no conditioning)."""
    z2 = (observed - expected) ** 2 / expected
    return float(stats.chi2.sf(z2.sum(), df=observed.size))


def test_homogeneous_inter_arrivals_are_exponential_and_counts_poisson():
    rng = np.random.default_rng(1)
    # One long window: gaps from many short windows would miss each window's censored last gap.
    long_run = poisson_arrivals(rng, 0.5, 0.0, 40_000.0)
    gaps = np.diff(np.concatenate([[0.0], long_run]))
    assert stats.kstest(gaps, stats.expon(scale=2.0).cdf).pvalue > ALPHA
    counts = []
    for _ in range(3000):
        times = poisson_arrivals(rng, 0.5, 0.0, 20.0)
        assert np.all((times >= 0) & (times < 20)) and np.all(np.diff(times) > 0)
        counts.append(times.size)
    values = np.arange(0, 31)
    observed = np.array([np.sum(np.asarray(counts) == v) for v in values])
    probs = stats.poisson.pmf(values, 10.0)
    probs[-1] += stats.poisson.sf(30, 10.0)
    observed[-1] += int(np.sum(np.asarray(counts) > 30))
    keep = probs * len(counts) >= 5
    merged_obs = np.append(observed[keep], observed[~keep].sum())
    merged_exp = np.append(probs[keep], probs[~keep].sum()) * len(counts)
    assert stats.chisquare(merged_obs, merged_exp).pvalue > ALPHA


def test_thinning_matches_the_rate_integral_per_period():
    breaks, rates = (0.0, 120.0, 300.0, 480.0), (0.2, 0.5, 0.1)
    rng = np.random.default_rng(7)
    per_period = np.zeros(3)
    n = 400
    for _ in range(n):
        times = thinned_arrivals(rng, breaks, rates, 0.0, 480.0)
        per_period += np.histogram(times, bins=breaks)[0]
    expected = np.array(
        [expected_count(breaks, rates, lo, hi) for lo, hi in itertools.pairwise(breaks)]
    )
    np.testing.assert_allclose(expected, [24.0, 90.0, 18.0])
    assert poisson_counts_pvalue(per_period, expected * n) > ALPHA
    # Within a period, arrivals are uniform.
    rng = np.random.default_rng(8)
    inside = np.concatenate(
        [
            t[(t >= 120) & (t < 300)]
            for t in (thinned_arrivals(rng, breaks, rates, 0, 480) for _ in range(200))
        ]
    )
    assert stats.kstest(inside, stats.uniform(120, 180).cdf).pvalue > ALPHA


def test_cutoff_truncates_and_rates_outside_the_profile_are_zero():
    breaks, rates = (10.0, 20.0), (5.0,)
    rng = np.random.default_rng(0)
    times = np.concatenate([thinned_arrivals(rng, breaks, rates, 0.0, 15.0) for _ in range(50)])
    assert times.min() >= 10.0 and times.max() < 15.0
    np.testing.assert_array_equal(
        piecewise_rate(breaks, rates, np.array([5.0, 10.0, 19.9, 20.0])), [0, 5, 5, 0]
    )
    assert expected_count(breaks, rates, 0.0, 15.0) == 25.0
    assert poisson_arrivals(rng, 0.0, 0.0, 10.0).size == 0
    assert thinned_arrivals(rng, (0.0, 1.0), (0.0,), 0.0, 1.0).size == 0


def test_dynamism_metrics_on_a_hand_computed_example():
    # Four requests: two known at the start, two released at 50 and 75 on [0, 100).
    release = [0.0, 0.0, 50.0, 75.0, np.inf]
    assert degree_of_dynamism(release, 0.0) == 0.5
    assert effective_degree_of_dynamism(release, 0.0, 100.0) == pytest.approx((0.5 + 0.75) / 4)
    assert degree_of_dynamism([], 0.0) == 0.0
    assert effective_degree_of_dynamism([np.inf], 0.0, 1.0) == 0.0
    with pytest.raises(ValueError, match="end > start"):
        effective_degree_of_dynamism(release, 5.0, 5.0)
