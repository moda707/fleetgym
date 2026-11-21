"""Recourse rules on a hand-made line instance with fixed scenarios."""

from __future__ import annotations

import numpy as np
import pytest

from fleetgym.data import Instance
from fleetgym.data.schema import register_schema
from fleetgym.envs.env import FleetGymEnv
from fleetgym.errors import ConfigError, InfeasibleActionError
from fleetgym.sim.decisions import Dispatch, RecourseChoice
from fleetgym.sim.engine import RecourseLoopError
from fleetgym.sim.recourse import RECOURSE_CLASSES, RecourseRule, register_recourse
from fleetgym.spec import load_spec
from fleetgym.spec.models import FrozenModel
from fleetgym.spec.registries import RECOURSE_RULES
from fleetgym.testing.non_anticipativity import check_non_anticipativity
from fleetgym.testing.validator import validate
from fleetgym.uncertainty.scenario import Scenario

# Depot at 0, customers at x = 1, 2, 3 on a line; capacity 4.
LINE = Instance(
    name="line",
    source="test",
    demands=np.array([0.0, 2.0, 2.0, 2.0]),
    coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
    capacity=4.0,
)
DEMAND = {"dist": "poisson", "mean": "from_instance", "revealed": "on_arrival"}


def spec(rule, *, objective=None, **extra):
    return load_spec(
        {
            "fleet": {"sizing": "unlimited", "types": [{"name": "v", "capacity": "from_instance"}]},
            "network": {"type": "euclidean", "rounding": "none"},
            "constraints": {"capacity": {}},
            "objective": objective or {"distance": 1.0, "outsourcing": 1.0},
            "decisions": {"interface": "next_node", "epochs": "sequential"},
            "uncertainty": {"demand": DEMAND},
            "recourse": {"capacity_failure": rule},
            **extra,
        }
    )


def run(s, demands, actions):
    env = FleetGymEnv(s, instance=LINE, debug=True)
    problem_hash = s.content_hash()
    scenario = Scenario(problem_hash, LINE.content_hash(), {"demand": np.array(demands, float)})
    env.reset(seed=0, options={"scenario": scenario})
    infos = []
    for action in actions:
        _, _, terminated, _, info = env.step(action)
        infos.append(info)
        if terminated:
            break
    report = validate(env.event_log, s, LINE, scenario=scenario)
    assert report.ok, report.violations
    return env, infos


def kinds(env, *wanted):
    return [(r["kind"], r.get("node")) for r in env.event_log.rows if r["kind"] in wanted]


def test_return_to_depot_serves_what_fits_and_comes_back():
    env, _ = run(spec("return_to_depot"), [0, 3, 3, 0], [1, 2, 3, 0])
    # 3 fits at 1; at 2 only 1 fits: serve 1, restock, come back for 2.
    rows = kinds(env, "service_start", "failure", "restock")
    assert rows == [
        ("service_start", 1),
        ("service_start", 2),
        ("failure", 2),
        ("restock", 0),
        ("service_start", 2),
        ("service_start", 3),
    ]
    served = [r["quantity"] for r in env.event_log.of_kind("service_start")]
    assert served == [3.0, 1.0, 2.0, 0.0]
    assert env.simulator.state.total_costs["distance"] == pytest.approx(1 + 1 + 2 * 2 + 1 + 3)


def test_partial_service_false_restocks_before_serving():
    rule = {"rule": "return_to_depot", "partial_service": False}
    env, _ = run(spec(rule), [0, 3, 3, 0], [1, 2, 3, 0])
    served = [(r["node"], r["quantity"]) for r in env.event_log.of_kind("service_start")]
    assert served == [(1, 3.0), (2, 3.0), (3, 0.0)]


def test_demand_above_capacity_needs_several_trips_even_without_partial_service():
    rule = {"rule": "return_to_depot", "partial_service": False}
    env, _ = run(spec(rule), [0, 9, 0, 0], [1, 2, 3, 0])
    served = [r["quantity"] for r in env.event_log.of_kind("service_start") if r["node"] == 1]
    assert served == [4.0, 4.0, 1.0]
    assert len(env.event_log.of_kind("restock")) == 2


def test_exact_depletion_restock_waits_at_the_depot_for_a_decision():
    rule = {"rule": "return_to_depot", "on_exact_depletion": "restock"}
    env, infos = run(spec(rule), [0, 4, 1, 0], [1, 2, 3, 0])
    rows = kinds(env, "service_start", "restock", "route_end")
    assert rows[:2] == [("service_start", 1), ("restock", 0)]
    assert infos[0]["vehicle"] == 0 and env.event_log.of_kind("route_end")
    # At the depot after the restock, choosing the depot ends the route.
    env2 = FleetGymEnv(spec(rule), instance=LINE)
    scenario = Scenario(
        spec(rule).content_hash(), LINE.content_hash(), {"demand": np.array([0, 4, 0, 0.0])}
    )
    env2.reset(seed=0, options={"scenario": scenario})
    env2.step(1)
    assert env2.simulator.state.vehicles[0].node == 0 and env2.action_masks()[0]
    env2.step(0)
    assert env2.simulator.state.vehicles[0].status.name == "DONE"
    assert validate(env2.event_log, spec(rule), LINE, scenario=scenario).ok


def test_outsource_prices_the_remainder():
    rule = {"rule": "outsource", "per_customer": 2.0, "per_unit": 0.5}
    env, _ = run(spec(rule), [0, 3, 3, 0], [1, 2, 3, 0])
    (row,) = env.event_log.of_kind("outsource")
    assert (row["node"], row["quantity"], row["amount"]) == (2, 2.0, 3.0)
    assert env.simulator.state.total_costs["outsourcing"] == 3.0
    assert env.simulator.state.outsourced[2] == 2.0
    full = {"rule": "outsource", "per_unit": 1.0, "partial_service": False}
    env, _ = run(spec(full), [0, 3, 3, 0], [1, 2, 3, 0])
    assert env.event_log.of_kind("outsource")[0]["quantity"] == 3.0


def test_outsourcing_cost_can_come_from_the_instance():
    inst = Instance(
        name="line-costs",
        source="test",
        demands=LINE.demands,
        coords=LINE.coords,
        capacity=4.0,
        extra={"costs": np.array([0.0, 10.0, 20.0, 30.0])},
        schema=register_schema(
            {"schema": "tests/line-costs/1", "extends": "fleetgym/core/1", "node": {"costs": {}}}
        ).id,
    )
    rule = {"rule": "outsource", "per_customer": {"from_instance": "costs"}}
    s = spec(rule)
    env = FleetGymEnv(s, instance=inst)
    scenario = Scenario(s.content_hash(), inst.content_hash(), {"demand": np.array([0, 3, 3, 0.0])})
    env.reset(seed=0, options={"scenario": scenario})
    for a in (1, 2, 3, 0):
        env.step(a)
    assert env.simulator.state.total_costs["outsourcing"] == 20.0
    assert validate(env.event_log, s, inst, scenario=scenario).ok


def test_agent_decides_opens_a_recourse_epoch():
    rule = {"rule": "agent_decides", "per_unit": 1.0}
    s = spec(rule)
    env, infos = run(s, [0, 3, 3, 3], [1, 2, 2, 3, 0, 0, 0])
    assert infos[1]["epoch_kind"] == "recourse"
    assert np.flatnonzero(infos[1]["action_mask"]).tolist() == [0, 2]
    assert env.event_log.of_kind("failure")[0]["component"] == "outsource"
    restock_only = {"rule": "agent_decides", "options": ["return_to_depot"]}
    env = FleetGymEnv(spec(restock_only), instance=LINE)
    scenario = Scenario(
        spec(restock_only).content_hash(), LINE.content_hash(), {"demand": np.array([0, 3, 3, 0.0])}
    )
    env.reset(seed=0, options={"scenario": scenario})
    env.step(1)
    _, _, _, _, info = env.step(2)
    assert np.flatnonzero(info["action_mask"]).tolist() == [0]
    with pytest.raises(InfeasibleActionError, match="recourse epoch"):
        env.step(2)
    sim = env.simulator
    with pytest.raises(InfeasibleActionError, match="not allowed"):
        sim.apply(RecourseChoice(0, "outsource"))
    with pytest.raises(InfeasibleActionError, match="RecourseChoice"):
        sim.apply(Dispatch(0, 3))
    env.step(0)
    with pytest.raises(InfeasibleActionError, match="no capacity failure"):
        sim.apply(RecourseChoice(0, "return_to_depot"))


class _CleverParams(FrozenModel):
    radius: float
    per_unit: float = 1.0
    per_customer: float = 0.0


class _Clever(RecourseRule):
    def on_failure(self, failure, info):
        near = info.distances[failure.node, 0] <= self.params.radius  # pyright: ignore[reportAttributeAccessIssue]
        return "return_to_depot" if near else "outsource"


class _Stubborn(RecourseRule):
    def serve_first(self, failure, info):
        return False

    def on_failure(self, failure, info):
        return "return_to_depot"


@pytest.fixture
def custom_rules():
    """Register the test rules for one test only (the registries are global)."""
    register_recourse("tests.clever", params=_CleverParams, description="test rule")(_Clever)
    register_recourse("tests.stubborn", params=_CleverParams)(_Stubborn)
    yield
    for name in ("tests.clever", "tests.stubborn"):
        RECOURSE_RULES.unregister(name)
        RECOURSE_CLASSES.unregister(name)


@pytest.mark.usefixtures("custom_rules")
def test_custom_rules_register_and_run():
    s = spec({"rule": "tests.clever", "radius": 1.5})
    env, _ = run(s, [0, 3, 3, 3], [1, 2, 3, 0])
    assert [r["component"] for r in env.event_log.of_kind("failure")] == ["outsource", "outsource"]
    near = spec({"rule": "tests.clever", "radius": 2.5})
    env, _ = run(near, [0, 3, 3, 3], [1, 2, 3, 0])
    assert [r["component"] for r in env.event_log.of_kind("failure")] == [
        "return_to_depot",
        "outsource",
    ]
    with pytest.raises(ConfigError, match="already registered"):
        register_recourse("tests.clever", params=_CleverParams)(_Clever)


@pytest.mark.usefixtures("custom_rules")
def test_a_rule_that_restocks_an_empty_vehicle_is_stopped():
    s = spec({"rule": "tests.stubborn", "radius": 0})
    env = FleetGymEnv(s, instance=LINE)
    scenario = Scenario(s.content_hash(), LINE.content_hash(), {"demand": np.array([0, 4, 3, 0.0])})
    env.reset(seed=0, options={"scenario": scenario})
    env.step(1)  # load 4: empty capacity 0
    env.step(2)  # failure, nothing served, restock; back with an empty truck it serves (guard)
    env.step(3)
    env.step(0)
    assert validate(env.event_log, s, LINE, scenario=scenario).ok
    assert [r["quantity"] for r in env.event_log.of_kind("service_start")] == [4.0, 3.0, 0.0]
    zero_cap = Scenario(s.content_hash(), LINE.content_hash(), {"demand": np.array([0, 0, 5, 0.0])})
    env.reset(seed=0, options={"scenario": zero_cap})
    env.step(2)  # empty vehicle: always serves 4 first, then restocks for the last unit
    assert [r["quantity"] for r in env.event_log.of_kind("service_start")][:1] == [4.0]
    sim = env.simulator
    v = sim.state.vehicles[0]
    v.load = 0.0
    with pytest.raises(RecourseLoopError):
        sim._resolve(v, 2, "return_to_depot")  # pyright: ignore[reportPrivateUsage]


def test_validator_catches_recourse_corruptions():
    s = spec("return_to_depot")
    env, _ = run(s, [0, 3, 3, 0], [1, 2, 3, 0])
    scenario = env.scenario
    assert scenario is not None

    def codes(mutate):
        log = env.event_log.copy()
        mutate(log.rows)
        return {v.code for v in validate(log, s, LINE, scenario=scenario).violations}

    def first(rows, kind):
        return next(r for r in rows if r["kind"] == kind)

    assert "quantity" in codes(lambda rows: first(rows, "service_start").update(quantity=2.0))
    assert "recourse" in codes(lambda rows: first(rows, "failure").update(component="outsource"))
    assert "recourse" in codes(lambda rows: first(rows, "failure").update(quantity=5.0))
    assert "restock" in codes(lambda rows: first(rows, "restock").update(time=99.0))
    assert "reveal" in codes(lambda rows: first(rows, "reveal").update(value=7.0))
    other = Scenario(s.content_hash(), LINE.content_hash(), {"demand": np.array([0, 3, 2, 0.0])})
    assert "identity" in {
        v.code for v in validate(env.event_log, s, LINE, scenario=other).violations
    }

    def drop_restock(rows):
        rows[:] = [r for r in rows if r["kind"] != "restock"]

    assert "depart" in codes(drop_restock)


def test_fork_redraws_only_hidden_values_and_clone_keeps_the_future():
    s = spec("return_to_depot")
    env = FleetGymEnv(s, instance=LINE)
    env.reset(seed=4)
    env.step(1)  # demand of node 1 revealed
    truth = env.scenario
    assert truth is not None
    forks = [env.fork(seed=k) for k in range(30)]
    for f in forks:
        assert f.scenario is not None
        assert f.scenario["demand"][1] == truth["demand"][1]
        assert f.simulator.state.revealed["demand"].tolist() == [True, True, False, False]
    assert any(f.scenario["demand"][2] != truth["demand"][2] for f in forks if f.scenario)
    assert env.clone().scenario == truth
    deterministic = load_spec("CVRP-v0")
    det = FleetGymEnv(deterministic, instance=LINE)
    det.reset(seed=0)
    assert det.fork(seed=1).scenario is None


def test_vrpsd_is_non_anticipative():
    import fleetgym as fg

    for seed in range(3):
        assert check_non_anticipativity(fg.make("VRPSD-v0"), seed=seed) > 0
    leaky = fg.make("VRPSD-v0")
    original = leaky._observation  # pyright: ignore[reportPrivateUsage]

    def peek():
        obs = original()
        obs["globals"] = obs["globals"].copy()
        obs["globals"][0] = float(leaky.simulator._truth["demand"].sum())  # pyright: ignore[reportPrivateUsage]
        return obs

    from fleetgym.testing.non_anticipativity import AnticipationError

    leaky._observation = peek  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AnticipationError):
        check_non_anticipativity(leaky, seed=0)


def _log_codes(s, scenario, rows):
    from fleetgym.eventlog import EventLog

    log = EventLog(dict(_LOG_META[0]))
    log.rows = rows
    return {v.code for v in validate(log, s, LINE, scenario=scenario).violations}


_LOG_META: list[dict] = []


def _corrupt(s, demands, actions, mutate):
    env, _ = run(s, demands, actions)
    _LOG_META[:] = [env.event_log.metadata]
    rows = [dict(r) for r in env.event_log.rows]
    mutate(rows)
    return _log_codes(s, env.scenario, rows)


def _first(rows, kind, **match):
    return next(
        r for r in rows if r["kind"] == kind and all(r.get(k) == v for k, v in match.items())
    )


OUTSOURCE = {"rule": "outsource", "per_customer": 2.0, "per_unit": 0.5}
RESTOCK_RULE = {"rule": "return_to_depot", "on_exact_depletion": "restock"}


@pytest.mark.parametrize(
    ("rule", "demands", "actions", "mutate", "code"),
    [
        (
            OUTSOURCE,
            [0, 3, 3, 0],
            [1, 2, 3, 0],
            lambda r: _first(r, "outsource").update(amount=9.0),
            "outsource",
        ),
        (
            OUTSOURCE,
            [0, 3, 3, 0],
            [1, 2, 3, 0],
            lambda r: _first(r, "outsource").update(quantity=1.0),
            "outsource",
        ),
        (
            "return_to_depot",
            [0, 3, 3, 0],
            [1, 2, 3, 0],
            lambda r: _first(r, "failure").update(load=0.0),
            "load",
        ),
        (
            "return_to_depot",
            [0, 3, 3, 0],
            [1, 2, 3, 0],
            lambda r: _first(r, "failure").update(node=1),
            "recourse",
        ),
        (
            "return_to_depot",
            [0, 3, 3, 0],
            [1, 2, 3, 0],
            lambda r: _first(r, "reveal").update(component="service_time"),
            "reveal",
        ),
        (
            "return_to_depot",
            [0, 3, 3, 0],
            [1, 2, 3, 0],
            lambda r: r.insert(5, dict(_first(r, "reveal"))),
            "reveal",
        ),
        (
            "return_to_depot",
            [0, 3, 3, 0],
            [1, 2, 3, 0],
            lambda r: r.append({"seq": 999, "kind": "unserved", "node": 1}),
            "unserved",
        ),
        (
            "return_to_depot",
            [0, 3, 3, 0],
            [1, 2, 3, 0],
            lambda r: r.__setitem__(
                slice(None), [x for x in r if not (x["kind"] == "service_start" and x["node"] == 3)]
            ),
            "service",
        ),
        (
            RESTOCK_RULE,
            [0, 4, 1, 0],
            [1, 2, 3, 0],
            lambda r: _first(r, "service_start", node=1).update(quantity=3.0, load=3.0),
            "recourse",
        ),
        (
            OUTSOURCE,
            [0, 3, 3, 0],
            [1, 2, 3, 0],
            lambda r: _first(r, "failure").update(component="return_to_depot"),
            "recourse",
        ),
    ],
)
def test_validator_catches_more_recourse_corruptions(rule, demands, actions, mutate, code):
    assert code in _corrupt(spec(rule), demands, actions, mutate)


def test_validator_rejects_failures_without_a_rule_and_undelivered_demand():
    s = spec("return_to_depot")
    env, _ = run(s, [0, 3, 3, 0], [1, 2, 3, 0])
    no_rule = load_spec(
        {
            **s.to_dict(),
            "recourse": {},
            "uncertainty": {"demand": {**DEMAND, "revealed": "a_priori"}},
        }
    )
    assert not validate(
        env.event_log,
        no_rule,
        LINE,
        scenario=Scenario(
            no_rule.content_hash(), LINE.content_hash(), {"demand": np.array([0, 3, 3, 0.0])}
        ),
    ).ok
    rows = [dict(r) for r in env.event_log.rows]
    second = [i for i, r in enumerate(rows) if r["kind"] == "service_start" and r["node"] == 2][1]
    rows[second]["quantity"] = 1.0
    _LOG_META[:] = [env.event_log.metadata]
    assert {"quantity", "load"} <= _log_codes(s, env.scenario, rows)
