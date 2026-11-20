"""Evaluation harness, metrics, statistics and report cards."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats

from fleetgym.baselines import NearestNeighbor, PerfectInformation, RandomPolicy
from fleetgym.errors import ValidationFailure
from fleetgym.eval import (
    compare,
    cvar,
    evaluate,
    mean_ci,
    report_card,
    scenario_seeds,
    summarize,
    write_report_card,
)

SMALL = {
    "extends": "CVRP-v0",
    "instance_source": {
        "type": "generator",
        "name": "uniform",
        "params": {"n_customers": 5, "size": 100, "capacity": 10, "demand": {"low": 1, "high": 5}},
    },
}


def test_scenario_seeds_are_fixed_and_disjoint():
    test = scenario_seeds("CVRP-v0", "test", 50)
    assert test == scenario_seeds("CVRP-v0", "test", 50)
    assert scenario_seeds("CVRP-v0", "test", 10) == test[:10]
    train = set(scenario_seeds("CVRP-v0", "train", 50))
    assert not train & set(test)
    assert scenario_seeds("VRPTW-v0", "test", 5) != test[:5]
    with pytest.raises(ValueError, match="positive"):
        scenario_seeds("CVRP-v0", "test", 0)
    with pytest.raises(ValueError, match="unknown split"):
        scenario_seeds("CVRP-v0", "dev", 3)  # pyright: ignore[reportArgumentType]


def test_statistics_agree_with_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 2, 40)
    mean, low, high = mean_ci(x.tolist())
    ref = stats.t.interval(0.95, x.size - 1, loc=x.mean(), scale=stats.sem(x))
    assert (low, high) == pytest.approx(ref)
    assert mean == pytest.approx(x.mean())
    assert mean_ci([3.0]) == (3.0, 3.0, 3.0)
    assert cvar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == 10
    assert cvar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.8) == 9.5
    y = x + rng.normal(0.5, 1, 40)
    result = compare(x.tolist(), y.tolist())
    assert result.t_pvalue == pytest.approx(stats.ttest_rel(x, y).pvalue)
    assert result.wilcoxon_pvalue == pytest.approx(stats.wilcoxon(x, y).pvalue)  # pyright: ignore[reportAttributeAccessIssue]
    assert result.mean_difference == pytest.approx(float(np.mean(x - y)))
    same = compare([1.0, 2.0], [1.0, 2.0])
    assert same.t_pvalue == same.wilcoxon_pvalue == 1.0
    with pytest.raises(ValueError, match="two pairs"):
        compare([1.0], [2.0])


def test_evaluate_and_report_card(tmp_path):
    seeds = scenario_seeds("small", "test", 6)
    nearest = evaluate(SMALL, NearestNeighbor, seeds)
    random = evaluate(SMALL, lambda: RandomPolicy(1), seeds)
    assert len(nearest.episodes) == 6 and nearest.policy == "nearest"
    summary = nearest.summary
    assert summary.ci_low <= summary.mean_cost <= summary.ci_high
    assert summary.cvar90 >= summary.mean_cost
    assert summarize(nearest.episodes) == summary
    card = report_card(nearest, reference=random)
    assert card["reference"]["policy"] == "random"
    assert card["reference"]["gap"] < 0  # nearest neighbor beats random
    assert card["spec_hash"].startswith("fg1:") and "numpy" in card["versions"]
    path = write_report_card(card, tmp_path / "cards" / "nearest.json")
    assert json.loads(path.read_text())["seeds"] == seeds
    oracle = evaluate(SMALL, PerfectInformation, seeds[:2])
    with pytest.raises(ValueError, match="bound"):
        report_card(oracle)
    assert report_card(oracle, bound=True)["kind"] == "perfect-information bound"
    with pytest.raises(ValueError, match="other seeds"):
        report_card(nearest, reference=oracle)


def test_evaluate_raises_on_invalid_episodes(monkeypatch):
    from fleetgym.eval import harness
    from fleetgym.testing.validator import ValidationReport, Violation

    monkeypatch.setattr(
        harness, "validate", lambda *a, **k: ValidationReport([Violation("x", "y")])
    )
    with pytest.raises(ValidationFailure, match="invalid episode"):
        evaluate(SMALL, NearestNeighbor, [0])
