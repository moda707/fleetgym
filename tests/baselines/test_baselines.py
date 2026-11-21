"""Baselines: valid trajectories, non-anticipativity, the oracle bound, rollout by hand."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import fleetgym as fg
from fleetgym.baselines import (
    AprioriRoute,
    CheapestInsertion,
    NearestNeighbor,
    PerfectInformation,
    RandomPolicy,
    Rollout,
    hindsight_optimum,
    run_episode,
)
from fleetgym.baselines.pyvrp_reopt import PyVRPReoptimization
from fleetgym.baselines.rollout import cost_to_go
from fleetgym.testing.non_anticipativity import check_policy_non_anticipative
from fleetgym.testing.validator import validate
from tests.families import instance_source, make_env, spec_for

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "baseline_costs.json"
DYNAMIC_NEXT = {
    "extends": "DVRP-v1",
    "instance_source": instance_source("dvrp"),
    "decisions": {"epochs": "on_event", "interface": "next_node"},
    "objective": {"distance": 1.0, "unserved": 50.0},
}
SMALL = {
    "extends": "CVRP-v1",
    "instance_source": instance_source("small"),
}
SMALL_SD = {"extends": "SVRP-v1", "instance_source": instance_source("small_sd")}
CONFIGS = [
    spec_for("CVRP-v1"),
    spec_for("VRPTW-v1"),
    spec_for("SVRP-v1"),
    spec_for("DVRP-v1"),
    DYNAMIC_NEXT,
]
IDS = ["CVRP", "VRPTW", "VRPSD", "DVRP-accept", "DVRP-next"]
POLICIES = {
    "random": lambda: RandomPolicy(seed=5),
    "nearest": NearestNeighbor,
    "insertion": CheapestInsertion,
}


def valid(env) -> bool:
    report = validate(env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario)
    return report.ok


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
@pytest.mark.parametrize("name", list(POLICIES))
def test_baselines_produce_valid_trajectories(config, name):
    env = fg.make(config)
    for seed in range(3):
        episode = run_episode(env, POLICIES[name](), seed=seed)
        assert valid(env)
        assert episode.cost == pytest.approx(
            sum(
                env.problem_spec.objective.scalar_weights().get(c, 0.0) * a
                for c, a in episode.costs.items()
            )
        )


@pytest.mark.extras_solvers
@pytest.mark.parametrize("config", [*CONFIGS[:3], DYNAMIC_NEXT], ids=[*IDS[:3], "DVRP-next"])
def test_pyvrp_reoptimization(config):
    pytest.importorskip("pyvrp")
    env = fg.make(config)
    policy = PyVRPReoptimization(iterations=150)
    episode = run_episode(env, policy, seed=1)
    pyvrp_cost = episode.cost
    assert valid(env)
    # PyVRP works in scaled integers: with time windows a suggested stop can be infeasible in
    # exact arithmetic, and the policy falls back (documented); it must stay rare.
    assert policy.fallbacks <= 0.1 * episode.steps
    nearest_cost = run_episode(env, NearestNeighbor(), seed=1).cost
    assert pyvrp_cost <= nearest_cost + 1e-9


def test_a_priori_route_and_rollout_are_valid():
    env = fg.make(SMALL_SD)
    run_episode(env, AprioriRoute([1, 2, 3, 4, 5]), seed=0)
    assert valid(env)
    run_episode(env, Rollout(NearestNeighbor, n_samples=2), seed=0)
    assert valid(env)


@pytest.mark.parametrize(
    ("config", "factory"),
    [
        (CONFIGS[2], NearestNeighbor),
        (CONFIGS[2], CheapestInsertion),
        (SMALL_SD, lambda: Rollout(NearestNeighbor, n_samples=2, seed=3)),
        (DYNAMIC_NEXT, NearestNeighbor),
        (CONFIGS[3], CheapestInsertion),
    ],
    ids=["nearest-sd", "insertion-sd", "rollout-sd", "nearest-dyn", "insertion-accept"],
)
def test_baselines_are_non_anticipative(config, factory):
    assert check_policy_non_anticipative(fg.make(config), factory, seed=1, n_forks=2) > 0


def test_perfect_information_is_not_non_anticipative():
    from fleetgym.testing.non_anticipativity import AnticipationError

    caught = 0
    for seed in range(5):
        try:
            check_policy_non_anticipative(fg.make(SMALL_SD), PerfectInformation, seed=seed)
        except AnticipationError:
            caught += 1
    assert caught > 0  # the oracle uses the future, and the check notices


@pytest.mark.parametrize("config", [SMALL, SMALL_SD], ids=["deterministic", "stochastic"])
def test_exact_hindsight_cost_bounds_every_non_anticipative_policy(config):
    env = fg.make(config)
    for seed in range(4):
        oracle = PerfectInformation()
        bound = run_episode(env, oracle, seed=seed).cost
        assert oracle.bound
        assert valid(env)
        for factory in (
            NearestNeighbor,
            CheapestInsertion,
            lambda seed=seed: RandomPolicy(seed),
            lambda: Rollout(NearestNeighbor, n_samples=2),
        ):
            assert bound <= run_episode(env, factory(), seed=seed).cost + 1e-9


def test_rollout_equals_the_hand_computed_improvement_step():
    env = fg.make(SMALL)
    obs, info = env.reset(seed=2)
    by_hand = {}
    for action in np.flatnonzero(info["action_mask"]):
        by_hand[int(action)] = cost_to_go(env.clone(), int(action), NearestNeighbor())
    best = min(by_hand, key=lambda a: (by_hand[a], a))
    rollout = Rollout(NearestNeighbor, n_samples=1)
    rollout.reset(env)
    assert rollout.act(obs, info) == best
    # The first rollout step can only improve on the base policy's own first step.
    base = NearestNeighbor()
    base.reset(env)
    assert by_hand[best] <= by_hand[base.act(obs, info)] + 1e-9


def test_hindsight_search_limits_and_heuristic_label():
    env = make_env("CVRP-v1")
    env.reset(seed=0)
    with pytest.raises(ValueError, match="limited"):
        hindsight_optimum(env)
    heuristic = PerfectInformation("rollout")
    assert not heuristic.bound and heuristic.anticipative
    small = fg.make(SMALL)
    assert (
        run_episode(small, heuristic, seed=0).cost
        >= run_episode(small, PerfectInformation(), seed=0).cost - 1e-9
    )


def test_baseline_costs_match_the_snapshot(update_golden):
    current = {}
    for config, label in zip(CONFIGS[:4], IDS[:4], strict=True):
        env = fg.make(config)
        for name, factory in POLICIES.items():
            current[f"{label}/{name}"] = round(run_episode(env, factory(), seed=11).cost, 6)
    if update_golden or not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    stored = json.loads(GOLDEN.read_text())
    changed = sorted(k for k in current if stored.get(k) != pytest.approx(current[k], rel=1e-9))
    assert not changed, (
        f"baseline costs changed for {changed}; rerun with --update-golden if intended"
    )
