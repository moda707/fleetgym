"""The EV shortest path with charging-station occupancy (M14, proposal 0003, ADR 0026)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy import stats

import fleetgym as fg
from fleetgym.baselines import run_episode
from fleetgym.baselines.ev import EVGreedy, EVSteadyState
from fleetgym.data import Instance
from fleetgym.envs.ev_env import EVShortestPathEnv
from fleetgym.errors import InfeasibleActionError
from fleetgym.sim.ev import charging_time
from fleetgym.spec import load_spec
from fleetgym.testing.strategies import random_masked_rollout
from fleetgym.testing.validator import validate
from fleetgym.uncertainty.queues import sample_stations, stationary_distribution, steady_state_wait
from tests.families import ev, first_instance, make_env


def corridor(
    n_stations=3, *, kappa=2, lam=0.36, mu=0.56, length=300.0, xs=None, battery=16.0
) -> Instance:
    """Origin (0, 0), destination (0, length), stations on the line at ``xs`` (km)."""
    xs = (
        np.linspace(length / (n_stations + 1), length * n_stations / (n_stations + 1), n_stations)
        if xs is None
        else np.asarray(xs)
    )
    n = len(xs) + 2
    station = np.arange(n) >= 2
    extra = {
        "battery_capacity": battery,
        "consumption": 0.125,
        "role": ["origin", "destination"] + ["station"] * len(xs),
        "technology": ["", ""] + ["normal"] * len(xs),
        "arrival_rate": np.where(station, lam, 0.0),
        "service_rate": np.where(station, mu, 0.0),
        "queue_capacity": np.where(station, kappa, 1),
    }
    for k, (level, time) in enumerate([(13.6, 0.62), (15.2, 0.77), (16.0, 1.01)], start=1):
        extra[f"charge_level_{k}"] = np.where(station, level, 0.0)
        extra[f"charge_time_{k}"] = np.where(station, time, 0.0)
    coords = np.vstack([[0.0, 0.0], [0.0, length], np.column_stack([np.zeros(len(xs)), xs])])
    return Instance(
        name="corridor",
        source="test",
        demands=np.zeros(n),
        coords=coords,
        vehicle_types=({"count": 1},),
        schema="fleetgym/evspp/1",
        extra=extra,
    )


def ev_env(instance: Instance, **queues) -> EVShortestPathEnv:
    spec = load_spec(
        {"extends": "EVSPP-v1", "ev": {"shortest_path": {"queues": queues}}}
        if queues
        else "EVSPP-v1"
    )
    return fg.make_ev(spec, instance=instance)


# ------------------------------------------------------------------ queues


def test_initial_queues_follow_the_stationary_distribution():
    inst = corridor(1, kappa=3, lam=0.73, mu=1.12)
    counts = [
        sample_stations(inst, seed, 1.0, "steady_state")["ev.initial"][2] for seed in range(4000)
    ]
    observed = np.bincount(np.asarray(counts, dtype=int), minlength=4)
    expected = stationary_distribution(0.73, 1.12, 3) * len(counts)
    assert stats.chisquare(observed, expected).pvalue > 1e-3
    assert stationary_distribution(0.36, 0.56, 1) == pytest.approx(
        [1 / (1 + 0.36 / 0.56), (0.36 / 0.56) / (1 + 0.36 / 0.56)]
    )
    assert steady_state_wait(0.36, 0.56, 1) == pytest.approx(
        (0.36 / 0.56) / (1 + 0.36 / 0.56) / 0.56
    )  # L / mu, eq. 11


def test_arrivals_are_poisson_and_stations_have_their_own_streams():
    inst = corridor(2, lam=0.5)
    n = [
        np.sum(sample_stations(inst, seed, 40.0, "empty")["ev.arrival_node"] == 2)
        for seed in range(800)
    ]
    assert np.mean(n) == pytest.approx(0.5 * 40.0, rel=0.03)
    a = sample_stations(inst, 7, 40.0, "steady_state")
    changed = dataclasses.replace(
        inst, extra={**inst.extra, "arrival_rate": np.array([0, 0, 0.5, 0.9])}
    )
    b = sample_stations(changed, 7, 40.0, "steady_state")
    first = a["ev.arrival"][a["ev.arrival_node"] == 2]
    assert np.array_equal(first, b["ev.arrival"][b["ev.arrival_node"] == 2])  # station 2 unchanged


def test_charging_time_is_piecewise_linear_through_the_breakpoints():
    levels, times = np.array([13.6, 15.2, 16.0]), np.array([0.62, 0.77, 1.01])
    assert charging_time(levels, times, 0.0) == 0.0
    assert charging_time(levels, times, 6.8) == pytest.approx(0.31)
    assert charging_time(levels, times, 15.2) == pytest.approx(0.77)
    assert charging_time(levels, times, 15.6) == pytest.approx(0.89)


# ------------------------------------------------------------------ simulator


def test_a_hand_computed_trip_without_other_evs():
    inst = corridor(1, lam=0.0, xs=[100.0], length=200.0)
    env = ev_env(inst, initial="empty")
    env.reset(seed=0)
    slots = env.slots
    env.step(2 * slots)  # to the station: 100 km, 1 h, 12.5 kWh
    assert env.simulator.state.battery == pytest.approx(16.0 - 12.5)
    full = 2 * slots + env.params.target_levels  # charge to 16 kWh
    env.step(full)
    _, _, terminated, *_ = env.step(1 * slots)  # to the destination
    assert terminated
    costs = env.simulator.state.total_costs
    assert costs["drive"] == pytest.approx(2.0)
    assert costs["charge"] == pytest.approx(
        1.01 - charging_time(np.array([13.6, 15.2, 16.0]), np.array([0.62, 0.77, 1.01]), 3.5)
    )
    assert "wait" not in costs
    assert env.simulator.state.time == pytest.approx(sum(costs.values()))
    assert validate(env.event_log, env.problem_spec, inst, scenario=env.scenario).ok


def test_a_full_station_cannot_be_joined_unless_allowed():
    inst = corridor(2, kappa=1, lam=0.0, xs=[100.0, 110.0], length=200.0)
    for full, allowed in (("leave", False), ("join", True)):
        env = ev_env(inst, initial="empty", full=full)
        env.reset(seed=0)
        sim = env.simulator
        sim.state.stations[2].in_service = ("x", 50.0)  # another EV charges at station 2
        sim._push(50.0, 0, 2, 0.0)  # pyright: ignore[reportPrivateUsage]
        sim.state.indicator[2] = 1.0
        env.step(2 * env.slots)
        stay = env.action_masks().reshape(-1, env.slots)[2, 1:]
        assert stay.any() == allowed


def test_diversion_starts_where_the_ev_is():
    env = ev(make_env("EVSPP-v1"))
    rng = np.random.default_rng(3)
    for seed in range(5):
        random_masked_rollout(env, rng, seed=seed)
        if env.event_log.of_kind("divert"):
            assert validate(
                env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario
            ).ok
            return
    pytest.fail("no diversion in five random episodes")


def test_invalid_actions_raise_or_are_repaired():
    inst = first_instance("EVSPP-v1")
    env = ev_env(inst)
    env.reset(seed=0)
    with pytest.raises(InfeasibleActionError):
        env.step(0)  # the origin is never a target
    repaired = fg.make_ev("EVSPP-v1", instance=inst, invalid_action="repair")
    repaired.reset(seed=0)
    repaired.step(0)


# ------------------------------------------------------------------ forks and validation


def test_forks_keep_what_the_ev_knows_and_redraw_the_rest():
    env = ev(make_env("EVSPP-v1"))
    obs, _ = env.reset(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(5):
        obs, *_ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
    futures = set()
    for seed in range(6):
        twin = env.fork(seed)
        tobs = twin._observation()  # pyright: ignore[reportPrivateUsage]
        for key in obs:
            np.testing.assert_array_equal(tobs[key], obs[key])
        random_masked_rollout_from_here(twin, seed)
        futures.add(round(twin.simulator.state.time, 6))
    assert len(futures) > 1


def random_masked_rollout_from_here(env, seed: int) -> None:
    rng = np.random.default_rng(seed)
    done = False
    while not done:
        _, _, done, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))


def test_the_validator_catches_a_tampered_log():
    env = ev(make_env("EVSPP-v1"))
    run_episode(env, EVSteadyState(), seed=1)
    log = env.event_log.copy()
    assert validate(log, env.problem_spec, env.base_instance, scenario=env.scenario).ok
    for row in log.rows:
        if row["kind"] == "charge_end":
            row["time"] -= 0.1  # charging shortened
            break
    report = validate(log, env.problem_spec, env.base_instance, scenario=env.scenario)
    assert any(v.code == "charge" for v in report.violations)


@pytest.mark.parametrize("policy", [EVGreedy, EVSteadyState])
def test_baselines_reach_the_destination_validly(policy):
    env = ev(make_env("EVSPP-v1"))
    for seed in range(4):
        episode = run_episode(env, policy(), seed=seed)
        assert env.simulator.state.terminated
        assert validate(
            env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario
        ).ok
        assert episode.cost == pytest.approx(env.simulator.state.time)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 10_000),
    n=st.integers(2, 6),
    kappa=st.integers(1, 3),
    lam=st.sampled_from([0.0, 0.22, 0.5, 1.01]),
    full=st.sampled_from(["leave", "join"]),
)
def test_random_rollouts_validate(seed, n, kappa, lam, full):
    rng = np.random.default_rng(seed)
    gaps = 300.0 / (n + 1)  # at most 100 km: always solvable with 128 km of range
    xs = gaps * np.arange(1, n + 1) + rng.uniform(-0.1, 0.1, n) * gaps  # gaps <= 120 km
    inst = corridor(n, kappa=kappa, lam=lam, xs=xs)
    env = ev_env(inst, full=full)
    total = random_masked_rollout(env, rng, seed=seed)
    report = validate(env.event_log, env.problem_spec, inst, scenario=env.scenario)
    assert report.ok, report.violations[:3]
    assert total == pytest.approx(-env.simulator.state.time)


def test_unsolvable_instances_are_rejected():
    from fleetgym.errors import InstanceError

    with pytest.raises(InstanceError, match="cannot be reached"):
        ev_env(corridor(2, xs=[97.0, 105.0])).reset(seed=0)
