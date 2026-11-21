"""Stochastic travel and service times (proposal 0002, ADR 0025).

Taş, Dellaert, van Woensel and de Kok (2012): travel time per unit distance Gamma(alpha, lambda),
soft windows without waiting. Errico et al. (2018): hard windows, discrete triangular service times.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import fleetgym as fg
from fleetgym.data import Instance
from fleetgym.errors import ConfigError, NotInScopeError
from fleetgym.spec import load_spec
from fleetgym.testing.non_anticipativity import check_non_anticipativity
from fleetgym.testing.strategies import random_masked_rollout
from fleetgym.testing.validator import validate
from fleetgym.uncertainty.distributions import build_distribution, gamma_arc_times, triangular_cdf

# ------------------------------------------------------------------ distributions


@pytest.mark.parametrize(
    ("low", "mode", "high"), [(8, 10, 12), (5, 7, 15), (45, 63, 135), (0, 0, 1)]
)
def test_triangular_cdf_matches_scipy(low, mode, high):
    x = np.linspace(low - 1, high + 1, 57)
    c = (mode - low) / (high - low)
    expected = stats.triang(c, loc=low, scale=high - low).cdf(x)
    np.testing.assert_allclose(triangular_cdf(x, low, mode, high), expected, atol=1e-12)


def test_discrete_triangular_puts_mass_on_every_grid_point():
    dist = build_distribution(
        "discrete_triangular", {"low": 8, "mode": 10, "high": 12, "step": 0.1}
    )
    support = dist.support(1)
    assert support is not None
    ((values, probs),) = support
    assert values.size == 41 and values[0] == 8.0 and values[-1] == pytest.approx(12.0)
    assert np.all(probs > 0) and probs.sum() == pytest.approx(1.0)
    assert dist.mean(1)[0] == pytest.approx(10.0, abs=1e-9)  # symmetric
    draws = dist.sample(np.random.default_rng(0), 50_000)
    assert np.mean(draws) == pytest.approx(10.0, abs=0.02)
    skewed = build_distribution(
        "discrete_triangular", {"low": 5, "mode": 7, "high": 15, "step": 0.1}
    )
    assert skewed.mean(1)[0] == pytest.approx((5 + 7 + 15) / 3, abs=0.01)


def test_discrete_triangular_parameters_are_checked():
    with pytest.raises(ConfigError, match="low <= mode <= high"):
        load_spec(
            {
                "uncertainty": {
                    "service_time": {
                        "dist": "discrete_triangular",
                        "low": 5,
                        "mode": 4,
                        "high": 6,
                        "step": 1,
                    }
                }
            }
        )
    with pytest.raises(ConfigError, match="whole number of steps"):
        load_spec(
            {
                "uncertainty": {
                    "service_time": {
                        "dist": "discrete_triangular",
                        "low": 0,
                        "mode": 1,
                        "high": 2,
                        "step": 0.3,
                    }
                }
            }
        )


def test_gamma_arc_times_follow_tas_eq_16_17():
    alpha, lam, d = 16.0, 0.0625, 7.5  # CV 0.25 per unit distance
    distances = np.array([[0.0, d], [d, 0.0]])
    rng = np.random.default_rng(1)
    draws = np.array(
        [
            gamma_arc_times({"shape": alpha, "scale": lam}, distances, rng)[0, 1]
            for _ in range(20_000)
        ]
    )
    assert stats.kstest(draws, stats.gamma(alpha * d, scale=lam).cdf).pvalue > 1e-3
    assert np.mean(draws) == pytest.approx(alpha * lam * d, rel=0.01)  # = d
    assert np.var(draws) == pytest.approx(alpha * lam**2 * d, rel=0.05)
    assert gamma_arc_times({"shape": alpha, "scale": lam}, distances, rng)[0, 0] == 0.0


# ------------------------------------------------------------------ Taş-like setting


def tas_instance(seed: int = 0, n: int = 6) -> Instance:
    rng = np.random.default_rng(seed)
    coords = np.vstack([[50.0, 50.0], rng.uniform(0, 100, (n, 2))])
    opens = rng.uniform(0, 150, n)
    return Instance(
        name=f"tas-{seed}",
        source="test",
        coords=coords,
        demands=np.concatenate([[0.0], rng.integers(1, 20, n)]),
        service_times=np.concatenate([[0.0], np.full(n, 10.0)]),
        time_windows=np.vstack([[0.0, 230.0], np.column_stack([opens, opens + 40])]),
        vehicle_types=({"capacity": 200.0},),
        extra={"max_route_duration": 480.0},
    )


def tas_spec(shape: float = 1.0, scale: float = 1.0, **changes) -> fg.ProblemSpec:
    data = {
        "network": {"type": "euclidean", "rounding": "none"},
        "fleet": {"sizing": "unlimited"},
        "constraints": {
            "capacity": {},
            "time_windows": {"mode": "soft", "early": "serve"},
            "route_duration": {"mode": "soft"},
        },
        "uncertainty": {
            "travel_time": {
                "dist": "gamma_per_distance",
                "shape": shape,
                "scale": scale,
                "revealed": "on_arrival",
            }
        },
        "decisions": {"interface": "next_node", "epochs": "sequential"},
        "objective": {
            "lateness": 1.0,
            "earliness": 0.1,
            "distance": 1.0,
            "vehicles": 400.0,
            "overtime": 5 / 6,
        },
    }
    data.update(changes)
    return load_spec(data)


@pytest.mark.parametrize("seed", range(5))
def test_stochastic_travel_episodes_validate_repeat_and_sum_to_the_cost(seed):
    spec, inst = tas_spec(), tas_instance(seed)
    env = fg.make(spec, instance=inst)
    total = random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
    report = validate(env.event_log, spec, inst, scenario=env.scenario)
    assert report.ok, report.violations[:3]
    assert report.costs == pytest.approx(env.simulator.state.total_costs)
    assert total == pytest.approx(-env.reward_model.cost(report.costs), rel=1e-9, abs=1e-9)
    digest = env.event_log.digest()
    random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
    assert env.event_log.digest() == digest


def test_arrivals_use_the_realized_arc_times_and_reveal_them():
    spec, inst = tas_spec(shape=0.0625, scale=16.0), tas_instance(3)  # CV 4
    env = fg.make(spec, instance=inst)
    random_masked_rollout(env, np.random.default_rng(0), seed=4)
    assert env.scenario is not None
    n = inst.n_nodes
    times = env.scenario["travel_time"].reshape(n, n)
    expected = env.simulator.network.time_matrix()
    departs = {(r["vehicle"], r["node"]): r for r in env.event_log.of_kind("depart")}
    gaps = []
    for row in env.event_log.of_kind("arrive"):
        dep = departs[(row["vehicle"], row["node"])]
        travel = row["time"] - dep["time"]
        assert travel == pytest.approx(times[dep["from_node"], row["node"]])
        gaps.append(abs(travel - expected[dep["from_node"], row["node"]]))
    assert max(gaps) > 1.0  # CV 4: realized times differ from the expected ones
    reveals = [r for r in env.event_log.of_kind("reveal") if r["component"] == "travel_time"]
    assert len(reveals) == len(env.event_log.of_kind("arrive"))


def test_stochastic_travel_is_non_anticipative():
    env = fg.make(tas_spec(shape=0.0625, scale=16.0), instance=tas_instance(5))
    assert check_non_anticipativity(env, seed=2, n_forks=3) > 0


def test_early_service_is_charged_as_earliness():
    inst = Instance(
        name="early",
        source="test",
        coords=np.array([[0.0, 0.0], [10.0, 0.0]]),
        demands=np.array([0.0, 1.0]),
        service_times=np.array([0.0, 5.0]),
        time_windows=np.array([[0.0, 1000.0], [50.0, 60.0]]),
        vehicle_types=({"capacity": 10.0},),
        extra={"max_route_duration": 480.0},
    )
    spec = load_spec(
        {
            "fleet": {"sizing": "unlimited"},
            "constraints": {"time_windows": {"mode": "soft", "early": "serve"}},
            "decisions": {"interface": "next_node", "epochs": "sequential"},
            "objective": {"distance": 1.0, "earliness": 1.0},
        }
    )
    env = fg.make(spec, instance=inst)
    env.reset(seed=0)
    env.step(1)
    while not env.simulator.state.terminated:
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert env.event_log.of_kind("service_start")[0]["time"] == 10.0  # no waiting
    assert env.simulator.state.total_costs["earliness"] == pytest.approx(40.0)
    assert validate(env.event_log, spec, inst).ok


def test_early_serve_needs_soft_windows():
    with pytest.raises(ConfigError, match="soft windows"):
        load_spec({"constraints": {"time_windows": {"early": "serve"}}})


def test_travel_time_scope_is_checked():
    inst = tas_instance(0)
    with pytest.raises(ConfigError, match="shape \\* scale"):
        fg.make(tas_spec(shape=2.0, scale=1.0), instance=inst).reset(seed=0)
    hard = tas_spec(constraints={"capacity": {}, "time_windows": {"mode": "hard"}})
    with pytest.raises(NotInScopeError, match="soft time limits"):
        fg.make(hard, instance=inst).reset(seed=0)
    other = tas_spec(
        uncertainty={
            "travel_time": {"dist": "gamma", "shape": 2, "mean": 1, "revealed": "on_arrival"}
        }
    )
    with pytest.raises(ConfigError, match="gamma_per_distance"):
        fg.make(other, instance=inst).reset(seed=0)


# ------------------------------------------------------------------ Errico-like setting


def errico_instance(seed: int = 0, n: int = 8) -> Instance:
    rng = np.random.default_rng(seed)
    coords = np.vstack([[35.0, 35.0], rng.uniform(0, 70, (n, 2))])
    opens = rng.uniform(0, 120, n)
    return Instance(
        name=f"errico-{seed}",
        source="test",
        coords=coords,
        demands=np.zeros(n + 1),
        service_times=np.concatenate([[0.0], np.full(n, 10.0)]),
        time_windows=np.vstack([[0.0, 200.0], np.column_stack([opens, opens + 30])]),
        vehicle_types=({"count": 4},),
        distributions={
            "service_time": {
                "dist": "discrete_triangular",
                "low": 5,
                "mode": 10,
                "high": 15,
                "step": 0.1,
            }
        },
    )


ERRICO = {
    "network": {"type": "euclidean", "rounding": "none"},
    "fleet": {"sizing": "limited"},
    "constraints": {"time_windows": {"mode": "hard", "depot": "ignore"}},
    "uncertainty": {"service_time": {"revealed": "on_completion"}},
    "decisions": {"interface": "next_node", "epochs": "on_event"},
    "objective": {"distance": 1.0, "unserved": 1000.0},
}


@pytest.mark.parametrize("seed", range(5))
def test_hard_windows_with_stochastic_service_never_start_late(seed):
    spec, inst = load_spec(ERRICO), errico_instance(seed)
    env = fg.make(spec, instance=inst)
    random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
    assert validate(env.event_log, spec, inst, scenario=env.scenario).ok
    assert inst.time_windows is not None
    for row in env.event_log.of_kind("service_start"):
        assert row["time"] <= inst.time_windows[row["node"], 1] + 1e-9


def test_stochastic_service_with_a_closing_depot_is_out_of_scope():
    inst = errico_instance(0)
    enforced = {**ERRICO, "constraints": {"time_windows": {"mode": "hard"}}}
    with pytest.raises(NotInScopeError, match="depot closing time"):
        fg.make(load_spec(enforced), instance=inst).reset(seed=0)


# ------------------------------------------------------------------ properties

import itertools  # noqa: E402

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 10_000),
    n=st.integers(1, 7),
    params=st.sampled_from([(16.0, 0.0625), (1.0, 1.0), (0.0625, 16.0)]),
    early=st.sampled_from(["wait", "serve"]),
)
def test_random_stochastic_travel_rollouts_validate(seed, n, params, early):
    shape, scale = params
    constraints = {
        "capacity": {},
        "time_windows": {"mode": "soft", "early": early},
        "route_duration": {"mode": "soft"},
    }
    spec, inst = tas_spec(shape, scale, constraints=constraints), tas_instance(seed, n)
    env = fg.make(spec, instance=inst)
    total = random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
    report = validate(env.event_log, spec, inst, scenario=env.scenario)
    assert report.ok, report.violations[:3]
    assert total == pytest.approx(-env.reward_model.cost(report.costs), rel=1e-9, abs=1e-9)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), n=st.integers(1, 8), width=st.sampled_from([10.0, 30.0, 80.0]))
def test_random_hard_window_rollouts_with_stochastic_service_validate(seed, n, width):
    inst = errico_instance(seed, n)
    assert inst.time_windows is not None
    tw = inst.time_windows.copy()
    tw[1:, 1] = tw[1:, 0] + width
    inst = Instance(**{**inst.__dict__, "time_windows": tw})
    spec = load_spec(ERRICO)
    env = fg.make(spec, instance=inst)
    random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
    assert validate(env.event_log, spec, inst, scenario=env.scenario).ok


def errico_success_probability(route, travel, windows, service_pmf):
    """Errico et al. (2018), eq. 14-20: exact success probability of one route.

    ``service_pmf`` maps a service time to its probability (the same at every customer).
    """
    first = route[0]
    pmf = {max(windows[first][0], travel[0][first]): 1.0}  # m-bar of eq. 20
    for prev, node in itertools.pairwise(route):
        arrival: dict[float, float] = {}
        for t, p in pmf.items():  # eq. 18: convolve with the service time and add travel
            for s, q in service_pmf.items():
                z = round(t + s + travel[prev][node], 9)
                arrival[z] = arrival.get(z, 0.0) + p * q
        a, b = windows[node]
        pmf = {}
        for z, p in arrival.items():  # eq. 15: truncate to [a, b]; later arrivals fail
            if z <= b + 1e-9:
                key = max(z, a)
                pmf[key] = pmf.get(key, 0.0) + p
    return sum(pmf.values())  # eq. 13


def test_simulated_success_rate_matches_erricos_recursion():
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]])
    windows = np.array([[0.0, 1000.0], [0.0, 30.0], [20.0, 26.0], [30.0, 42.0], [50.0, 58.0]])
    inst = Instance(
        name="line",
        source="test",
        coords=coords,
        demands=np.zeros(5),
        service_times=np.array([0.0, 5.0, 5.0, 5.0, 5.0]),
        time_windows=windows,
        vehicle_types=({"count": 1},),
        distributions={
            "service_time": {
                "dist": "discrete_triangular",
                "low": 2,
                "mode": 5,
                "high": 9,
                "step": 1,
            }
        },
    )
    spec = load_spec({**ERRICO, "fleet": {"sizing": "fixed"}})
    env = fg.make(spec, instance=inst)
    env.reset(seed=0)
    dist = build_distribution("discrete_triangular", {"low": 2, "mode": 5, "high": 9, "step": 1})
    support = dist.support(1)
    assert support is not None
    ((values, probs),) = support
    travel = env.simulator.network.time_matrix()
    route = [1, 2, 3, 4]
    exact = errico_success_probability(
        route, travel, windows, dict(zip(values.tolist(), probs.tolist(), strict=True))
    )
    assert 0.2 < exact < 0.95  # the test instance is neither trivial nor hopeless
    episodes, successes = 3000, 0
    for seed in range(episodes):
        env.reset(seed=seed)
        todo = list(route)
        done = False
        while not done:
            valid = set(np.flatnonzero(env.action_masks()).tolist())
            action = 0
            while todo:
                node = todo.pop(0)
                if node in valid:
                    action = node
                    break
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        successes += env.simulator.state.total_costs.get("unserved", 0.0) == 0
    rate = successes / episodes
    stderr = np.sqrt(exact * (1 - exact) / episodes)
    assert abs(rate - exact) < 4 * stderr, (rate, exact)
