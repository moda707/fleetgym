"""Hypothesis strategies for specs, instances and action sequences.

Requires the ``hypothesis`` package (in the ``dev`` extra). Used by FleetGym's own property tests
and available for testing extensions::

    from hypothesis import given
    from fleetgym.testing.strategies import problems

    @given(problems())
    def test_my_policy_never_crashes(problem):
        spec, instance = problem
        ...
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fleetgym.data.instance import Instance
from fleetgym.envs.env import FleetGymEnv
from fleetgym.spec import ProblemSpec, load_spec

try:
    from hypothesis import strategies as st
except ImportError as exc:  # pragma: no cover - dev dependency
    raise ImportError(
        "fleetgym.testing.strategies needs hypothesis: pip install fleetgym[dev]"
    ) from exc

__all__ = [
    "dynamic_problems",
    "instances",
    "problems",
    "random_masked_rollout",
    "spec_dicts",
    "stochastic_problems",
    "stochastic_spec_dicts",
]


@st.composite
def instances(
    draw: st.DrawFn,
    *,
    min_customers: int = 1,
    max_customers: int = 6,
    time_windows: bool | None = None,
) -> Instance:
    """Small random instances on an integer grid, with optional time windows and service times.

    Instances may be infeasible on purpose (demand above capacity, unreachable windows), so that
    unserved requests and dead ends are exercised too.
    """
    n = draw(st.integers(min_customers, max_customers)) + 1
    coords = np.array(
        draw(st.lists(st.tuples(st.integers(0, 20), st.integers(0, 20)), min_size=n, max_size=n)),
        dtype=float,
    )
    demands = np.array(
        [0, *draw(st.lists(st.integers(0, 6), min_size=n - 1, max_size=n - 1))], dtype=float
    )
    capacity = float(draw(st.integers(3, 15)))
    with_tw = draw(st.booleans()) if time_windows is None else time_windows
    windows = service = None
    if with_tw:
        horizon = float(draw(st.integers(40, 200)))
        opens = draw(st.lists(st.integers(0, 100), min_size=n - 1, max_size=n - 1))
        widths = draw(st.lists(st.integers(0, 80), min_size=n - 1, max_size=n - 1))
        windows = np.array(
            [[0.0, horizon]] + [[o, o + w] for o, w in zip(opens, widths, strict=True)], dtype=float
        )
        service = np.array(
            [0, *draw(st.lists(st.integers(0, 10), min_size=n - 1, max_size=n - 1))], dtype=float
        )
    return Instance(
        name=f"hypothesis-n{n}",
        source="hypothesis",
        demands=demands,
        coords=coords,
        capacity=capacity,
        n_vehicles=draw(st.integers(1, 4)),
        time_windows=windows,
        service_times=service,
    )


@st.composite
def spec_dicts(draw: st.DrawFn, *, time_windows: bool) -> dict[str, Any]:
    """Random valid static spec dicts for instances with (or without) time windows."""
    epochs = draw(st.sampled_from(["sequential", "on_event"]))
    sizing = (
        "unlimited"
        if epochs == "sequential" and draw(st.booleans())
        else draw(st.sampled_from(["fixed", "limited"]))
    )
    vehicle: dict[str, Any] = {"name": "v", "capacity": "from_instance"}
    if sizing != "unlimited":
        vehicle["count"] = draw(st.one_of(st.integers(1, 4), st.just("from_instance")))
    constraints: dict[str, Any] = {}
    if draw(st.booleans()):
        constraints["capacity"] = {}
    if time_windows and draw(st.booleans()):
        constraints["time_windows"] = {
            "soft": draw(st.booleans()),
            "waiting": draw(st.sampled_from(["allowed", "forbidden"])),
        }
    if draw(st.booleans()):
        constraints["route_duration"] = {"max": float(draw(st.integers(20, 300)))}
    objective = draw(
        st.sampled_from(
            [
                {"distance": 1.0, "unserved": 100.0},
                {"distance": 1.0, "vehicles": 10.0, "unserved": 1000.0, "lateness": 2.0},
                {"lexicographic": ["unserved", "vehicles", "distance"]},
            ]
        )
    )
    return {
        "fleet": {"sizing": sizing, "types": [vehicle]},
        "network": {
            "type": "euclidean",
            "rounding": draw(st.sampled_from(["none", "nearest_int"])),
        },
        "constraints": constraints,
        "decisions": {"interface": "next_node", "epochs": epochs},
        "objective": objective,
    }


@st.composite
def problems(draw: st.DrawFn, *, max_customers: int = 6) -> tuple[ProblemSpec, Instance]:
    """A random ``(spec, instance)`` pair that binds."""
    instance = draw(instances(max_customers=max_customers))
    spec = load_spec(draw(spec_dicts(time_windows=instance.time_windows is not None)))
    return spec, instance


@st.composite
def stochastic_spec_dicts(draw: st.DrawFn) -> dict[str, Any]:
    """Random valid specs with uncertain demands (and service times) and a recourse rule.

    Time constraints only with the ``split`` rule (a route-duration limit, deterministic service
    times): other rules with time windows or duration limits are not in this version.
    """
    epochs = draw(st.sampled_from(["sequential", "on_event"]))
    sizing = (
        "unlimited"
        if epochs == "sequential" and draw(st.booleans())
        else draw(st.sampled_from(["fixed", "limited"]))
    )
    vehicle: dict[str, Any] = {"name": "v", "capacity": "from_instance"}
    if sizing != "unlimited":
        vehicle["count"] = draw(st.integers(1, 3))
    demand = draw(
        st.sampled_from(
            [
                {"dist": "poisson", "mean": "from_instance"},
                {
                    "dist": "categorical",
                    "values": [0, 1, 3, 7, 16],
                    "probs": [0.1, 0.3, 0.3, 0.2, 0.1],
                },
                {"dist": "fixed", "value": "from_instance"},
            ]
        )
    )
    demand["revealed"] = draw(st.sampled_from(["a_priori", "on_departure", "on_arrival"]))
    demand["forecast"] = draw(st.sampled_from(["mean", "none"]))
    uncertainty: dict[str, Any] = {"demand": demand}
    if draw(st.booleans()):
        uncertainty["service_time"] = {
            "dist": "exponential",
            "mean": 2.0,
            "revealed": draw(st.sampled_from(["on_arrival", "on_completion"])),
        }
    rule = draw(
        st.sampled_from(
            [
                "return_to_depot",
                {"rule": "return_to_depot", "partial_service": False},
                {"rule": "return_to_depot", "on_exact_depletion": "restock"},
                {"rule": "outsource", "per_unit": 2.0, "per_customer": 1.0},
                {"rule": "outsource", "per_unit": 1.0, "partial_service": False},
                {"rule": "agent_decides", "per_unit": 3.0},
                {"rule": "agent_decides", "options": ["return_to_depot"]},
                "split",
            ]
        )
    )
    constraints: dict[str, Any] = {"capacity": {}}
    objective: dict[str, float] = {"distance": 1.0, "outsourcing": 1.0, "unserved": 100.0}
    split = rule == "split"
    fleet: dict[str, Any] = {"sizing": sizing, "types": [vehicle]}
    if split:
        fleet["multi_trip"] = True
        fleet["end_route"] = draw(st.sampled_from(["anytime", "when_nothing_reachable"]))
        if draw(st.booleans()):
            fleet["reload_time"] = float(draw(st.integers(0, 5)))
        if draw(st.booleans()):
            uncertainty.pop("service_time", None)  # stochastic service times need time recourse
            constraints["route_duration"] = {
                "mode": draw(st.sampled_from(["hard", "soft"])),
                "max": float(draw(st.integers(20, 120))),
            }
        if draw(st.booleans()):
            objective = {"served_demand": -1.0, "distance": draw(st.sampled_from([0.0, 0.1]))}
        if constraints.get("route_duration", {}).get("mode") == "soft":
            objective = {**objective, "overtime": 1.0}
    return {
        "fleet": fleet,
        "network": {
            "type": "euclidean",
            "rounding": draw(st.sampled_from(["none", "nearest_int"])),
        },
        "constraints": constraints,
        "uncertainty": uncertainty,
        "recourse": {"capacity_failure": rule},
        "decisions": {"interface": "next_node", "epochs": epochs},
        "objective": objective,
    }


@st.composite
def stochastic_problems(draw: st.DrawFn, *, max_customers: int = 6) -> tuple[ProblemSpec, Instance]:
    """A random stochastic ``(spec, instance)`` pair that binds."""
    instance = draw(instances(max_customers=max_customers, time_windows=False))
    return load_spec(draw(stochastic_spec_dicts())), instance


@st.composite
def dynamic_problems(draw: st.DrawFn) -> tuple[ProblemSpec, Instance]:
    """A random dynamic-request ``(spec, instance)`` pair (accept_reject, assign or next_node)."""
    instance = draw(instances(max_customers=5))
    with_tw = instance.time_windows is not None
    constraints: dict[str, Any] = {"capacity": {}}
    if with_tw:
        constraints["time_windows"] = {"soft": draw(st.booleans())}
    if draw(st.booleans()):
        constraints["route_duration"] = {"max": float(draw(st.integers(30, 300)))}
    request_class: dict[str, Any] = {
        "location": {"dist": "uniform", "low": 0, "high": 20},
        "demand": {"dist": "categorical", "values": [0, 1, 3, 6], "probs": [0.1, 0.4, 0.3, 0.2]},
        "service_time": {"dist": "exponential", "mean": 2.0},
    }
    if with_tw and draw(st.booleans()):
        request_class["time_window_width"] = {"dist": "uniform", "low": 5, "high": 60}
    dynamics: dict[str, Any] = {
        "arrivals": {"process": "poisson", "rate": draw(st.sampled_from([0.05, 0.2, 0.5]))},
        "cutoff": float(draw(st.integers(10, 80))),
        "max_requests": draw(st.integers(1, 12)),
        "overflow": "reject",
        "initial_requests": draw(st.sampled_from(["instance", "mandatory", "none"])),
        "request_class": request_class,
    }
    if dynamics["initial_requests"] == "mandatory":
        dynamics["initial_infeasible"] = draw(st.sampled_from(["error", "serve_all"]))
    if draw(st.booleans()):
        dynamics["cancellations"] = {"dist": "bernoulli", "p": 0.3}
    interface = draw(st.sampled_from(["accept_reject", "assign", "next_node"]))
    decisions: dict[str, Any] = {"epochs": "on_event", "interface": interface}
    if interface != "next_node":
        decisions["routing"] = "cheapest_insertion"
        decisions["request_epochs"] = draw(st.sampled_from(["on_release", "on_vehicle_arrival"]))
    spec = {
        "fleet": {
            "sizing": draw(st.sampled_from(["fixed", "limited"])),
            "types": [{"name": "v", "capacity": "from_instance", "count": draw(st.integers(1, 3))}],
        },
        "network": {
            "type": "euclidean",
            "rounding": draw(st.sampled_from(["none", "nearest_int"])),
            "time_rounding": draw(st.sampled_from(["none", "ceil"])),
        },
        "constraints": constraints,
        "dynamics": dynamics,
        "decisions": decisions,
        "objective": {"distance": 1.0, "rejected": 20.0, "lateness": 1.0, "unserved": 100.0},
    }
    return load_spec(spec), instance


def random_masked_rollout(
    env: FleetGymEnv, rng: np.random.Generator, *, seed: int = 0, max_steps: int = 10_000
) -> float:
    """Play uniformly random valid actions until the episode ends; return the episode return.

    With ``initial_requests: mandatory``, `fleetgym.errors.InstanceError` is raised by the reset
    when the requests known at the start do not fit (random instances may be infeasible).

    Raises
    ------
    RuntimeError
        If the episode does not end within ``max_steps`` (a livelock).
    """
    env.reset(seed=seed)
    total = 0.0
    for _ in range(max_steps):
        valid = np.flatnonzero(env.action_masks())
        _, reward, terminated, truncated, _ = env.step(int(rng.choice(valid)))
        total += float(reward)
        if terminated or truncated:
            return total
    raise RuntimeError(f"episode did not end within {max_steps} steps")
