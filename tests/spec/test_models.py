from types import MappingProxyType

import pytest
from pydantic import ValidationError

from fleetgym.spec.models import (
    DistSpec,
    InstanceRef,
    ObjectiveSpec,
    ProblemSpec,
    RecourseRuleSpec,
    UncertainSpec,
)
from fleetgym.spec.models.base import deep_freeze, deep_thaw
from tests.spec.helpers import spec_dict


def make(**changes) -> ProblemSpec:
    return ProblemSpec.model_validate(spec_dict(**changes))


def errors_of(exc: pytest.ExceptionInfo[ValidationError]) -> list[tuple[tuple, str]]:
    return [(tuple(e["loc"]), e["msg"]) for e in exc.value.errors()]


# ------------------------------------------------------------------ basics and immutability


def test_minimal_spec_has_documented_defaults():
    spec = make()
    assert spec.network.name == "euclidean"
    assert spec.network.params.model_dump() == {
        "rounding": "none",
        "speed": 1.0,
        "time_rounding": "none",
        "time_model": None,
    }
    assert spec.decisions.interface == "next_node"
    assert spec.decisions.epochs == "on_event"
    assert spec.fleet.sizing == "unlimited"
    assert spec.meta.status == "experimental"
    assert spec.dynamics is None
    assert spec.agents.mode == "single"


def test_spec_is_deeply_immutable():
    spec = make(
        uncertainty={"demand": {"dist": "poisson", "mean": 3, "revealed": "on_arrival"}},
        meta={"instances": {"names": ["a", "b"]}},
    )
    with pytest.raises(ValidationError):
        spec.id = "X-v0"  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError):
        spec.uncertainty["x"] = spec.uncertainty["demand"]  # pyright: ignore[reportIndexIssue]
    with pytest.raises(TypeError):
        spec.constraints.items["x"] = spec.constraints["capacity"]  # pyright: ignore[reportIndexIssue]
    with pytest.raises(TypeError):
        spec.objective.weights["x"] = 1.0  # pyright: ignore[reportIndexIssue]
    with pytest.raises(TypeError):
        spec.meta.instances["names"] = ()  # pyright: ignore[reportIndexIssue]
    assert isinstance(spec.meta.instances["names"], tuple)
    assert isinstance(spec.fleet.types, tuple)
    with pytest.raises(ValidationError):
        spec.fleet.types[0].capacity = 5  # pyright: ignore[reportAttributeAccessIssue]


def test_unknown_keys_are_rejected_everywhere():
    with pytest.raises(ValidationError, match="Extra inputs"):
        make(bogus=1)
    with pytest.raises(ValidationError, match="Extra inputs"):
        make(fleet={"types": [{"name": "t", "capacty": 3}]})
    with pytest.raises(ValidationError, match="Extra inputs"):
        make(constraints={"time_windows": {"sotf": True}})


def test_deep_freeze_and_thaw_round_trip():
    data = {"a": [1, {"b": [2]}], "c": "x"}
    frozen = deep_freeze(data)
    assert isinstance(frozen, MappingProxyType)
    assert frozen["a"][1]["b"] == (2,)
    assert deep_thaw(frozen) == data


# ------------------------------------------------------------------ ids


@pytest.mark.parametrize("env_id", ["CVRP-v0", "DVRP-StochReq-v12", "a_b.c-v1"])
def test_valid_ids(env_id):
    assert make(id=env_id).id == env_id


@pytest.mark.parametrize("env_id", ["CVRP", "CVRP-v", "0CVRP-v0", "CVRP v0-v0", "CVRP-vx"])
def test_invalid_ids(env_id):
    with pytest.raises(ValidationError, match="environment IDs"):
        make(id=env_id)


# ------------------------------------------------------------------ instance references


@pytest.mark.parametrize(
    ("raw", "key", "dumped"),
    [
        ("from_instance", None, "from_instance"),
        ({"from_instance": "demands"}, "demands", {"from_instance": "demands"}),
    ],
)
def test_instance_ref_forms(raw, key, dumped):
    ref = InstanceRef.model_validate(raw)
    assert ref.key == key
    assert ref.model_dump() == dumped
    assert InstanceRef.model_validate(ref) is ref


@pytest.mark.parametrize(
    "raw", ["from_instanse", {"from_instance": ""}, {"from_instance": 3}, {"a": "b"}, 4]
)
def test_invalid_instance_refs(raw):
    with pytest.raises(ValidationError):
        InstanceRef.model_validate(raw)


def test_number_or_ref_fields():
    spec = make(fleet={"types": [{"name": "t", "capacity": "from_instance", "count": None}]})
    assert spec.fleet.types[0].capacity == InstanceRef(key=None)
    assert make().fleet.types[0].capacity == 100.0
    with pytest.raises(ValidationError):
        make(fleet={"types": [{"name": "t", "capacity": -1}]})


# ------------------------------------------------------------------ fleet


def test_fleet_sizing_rules():
    with pytest.raises(ValidationError, match="`count` is required with fixed sizing"):
        make(fleet={"sizing": "fixed", "types": [{"name": "t", "capacity": 1}]})
    with pytest.raises(ValidationError, match="not allowed with unlimited"):
        make(fleet={"types": [{"name": "t", "capacity": 1, "count": 2}]})
    with pytest.raises(ValidationError, match="unique"):
        make(fleet={"types": [{"name": "t"}, {"name": "t"}]})
    with pytest.raises(ValidationError, match="start < end"):
        make(fleet={"types": [{"name": "t"}], "shift": [480, 0]})
    assert make(fleet={"types": []}).fleet.from_instance  # empty: the instance's types
    fixed = make(fleet={"sizing": "limited", "types": [{"name": "t", "count": "from_instance"}]})
    assert fixed.fleet.types[0].count == InstanceRef()


# ------------------------------------------------------------------ distributions and uncertainty


@pytest.mark.parametrize(
    "raw",
    [
        {"dist": "fixed", "value": 3},
        {"dist": "uniform", "low": 0, "high": 1},
        {"dist": "normal", "mean": 0, "std": 1},
        {"dist": "truncated_normal", "mean": 0, "std": 1, "low": 0},
        {"dist": "lognormal", "mu": 2.3, "sigma": 0.4},
        {"dist": "gamma", "shape": 2, "mean": "from_instance"},
        {"dist": "weibull", "shape": 1.5, "scale": 3},
        {"dist": "exponential", "mean": 12},
        {"dist": "poisson", "mean": 0},
        {"dist": "bernoulli", "p": 0.02},
        {"dist": "categorical", "values": [5, 10, 20], "probs": [0.5, 0.3, 0.2]},
        {"dist": "empirical", "values": [1, 2, 3], "weights": [1, 0, 1]},
        {"dist": "empirical", "values": "from_instance"},
        {"dist": "kde", "data": "from_instance"},
        {"dist": "kde", "data": [[0, 0], [1, 1]], "bandwidth": 0.5},
    ],
)
def test_valid_distributions_round_trip(raw):
    dist = DistSpec.model_validate(raw)
    assert dist.name == raw["dist"]
    again = DistSpec.model_validate(dist.model_dump())
    assert again == dist


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"dist": "poison", "mean": 1}, "unknown distribution 'poison'"),
        ({"dist": "uniform", "low": 2, "high": 1}, "smaller than"),
        ({"dist": "normal", "mean": 0, "std": 0}, "greater than 0"),
        ({"dist": "truncated_normal", "mean": 0, "std": 1}, "at least one"),
        ({"dist": "truncated_normal", "mean": 0, "std": 1, "low": 2, "high": 1}, "smaller than"),
        ({"dist": "categorical", "values": [1, 2], "probs": [0.5, 0.4]}, "sum to 1"),
        ({"dist": "categorical", "values": [1, 2], "probs": [1.0]}, "entries"),
        ({"dist": "categorical", "values": [1, 2], "probs": [1.5, -0.5]}, "non-negative"),
        ({"dist": "bernoulli", "p": 1.2}, "less than or equal"),
        ({"dist": "empirical", "values": [1, 2], "weights": [1]}, "one per value"),
        ({"dist": "empirical", "values": [1, 2], "weights": [0, 0]}, "positive"),
        ({"dist": "empirical", "values": "from_instance", "weights": [1]}, "cannot be combined"),
        ({"dist": "poisson"}, "Field required"),
        ({"dist": "poisson", "mean": 1, "extra": 2}, "Extra inputs"),
        ({"dist": 3}, "must be a string"),
        ({"mean": 3}, "missing `dist`"),
    ],
)
def test_invalid_distributions(raw, match):
    with pytest.raises(ValidationError, match=match):
        DistSpec.model_validate(raw)


def test_nested_distribution_errors_keep_their_location():
    with pytest.raises(ValidationError) as exc:
        make(uncertainty={"demand": {"dist": "poisson", "mean": "abc", "revealed": "on_arrival"}})
    locs = [loc for loc, _ in errors_of(exc)]
    assert any(loc[:3] == ("uncertainty", "demand", "mean") for loc in locs), locs


def test_uncertain_spec_flat_form():
    raw = {"dist": "poisson", "mean": "from_instance", "revealed": "on_arrival"}
    unc = UncertainSpec.model_validate(raw)
    assert unc.revealed == "on_arrival"
    assert unc.dist is not None and unc.dist.name == "poisson"
    dumped = unc.model_dump()
    assert dumped == {**raw, "correlated_with": [], "forecast": "mean"}
    assert UncertainSpec.model_validate(dumped) == unc
    from_instance = UncertainSpec.model_validate({"revealed": "on_arrival"})
    assert from_instance.dist is None
    assert from_instance.model_dump() == {
        "revealed": "on_arrival",
        "correlated_with": [],
        "forecast": "mean",
    }
    with pytest.raises(ValidationError, match="revealed"):
        UncertainSpec.model_validate({"dist": "poisson", "mean": 1})
    with pytest.raises(ValidationError):
        UncertainSpec.model_validate({"dist": "poisson", "mean": 1, "revealed": "tomorrow"})


def test_unknown_and_reserved_uncertainty_sources():
    with pytest.raises(ValidationError, match="unknown uncertainty source"):
        make(uncertainty={"weather": {"dist": "fixed", "value": 1, "revealed": "a_priori"}})
    with pytest.raises(ValidationError, match=r"uncertainty\.presence"):
        make(uncertainty={"presence": {"dist": "fixed", "value": 1, "revealed": "a_priori"}})
    planned = make(
        meta={"status": "planned"},
        uncertainty={"presence": {"dist": "fixed", "value": 1, "revealed": "a_priori"}},
    )
    assert planned.reserved_features() == ("uncertainty.presence",)


def test_correlated_with_is_reserved():
    unc = {"dist": "fixed", "value": 1, "revealed": "a_priori", "correlated_with": ["weather"]}
    with pytest.raises(ValidationError, match="correlated_with"):
        make(uncertainty={"demand": unc})


# ------------------------------------------------------------------ constraints


def test_constraints_are_keyed_plugins():
    spec = make(
        constraints={
            "capacity": {},
            "time_windows": {"soft": True},
            "route_duration": {"max": "depot_due_time"},
            "outsourcing": {"cost": "per_customer", "value": "from_instance"},
        }
    )
    assert spec.constraints.names() == ("capacity", "time_windows", "route_duration", "outsourcing")
    assert "capacity" in spec.constraints
    assert spec.constraints["time_windows"].model_dump() == {
        "mode": "soft",
        "waiting": "allowed",
        "early": "wait",
        "depot": "enforce",
    }
    assert spec.to_dict()["constraints"]["route_duration"] == {
        "mode": "hard",
        "max": "depot_due_time",
    }
    default = make(constraints={"route_duration": {"mode": "soft"}})
    assert default.to_dict()["constraints"]["route_duration"] == {
        "mode": "soft",
        "max": "from_instance",
    }
    with pytest.raises(ValidationError, match="not both"):
        make(constraints={"time_windows": {"soft": True, "mode": "soft"}})


def test_constraint_null_params_mean_defaults():
    assert make(constraints={"capacity": None}).constraints["capacity"].model_dump() == {}


def test_constraint_errors():
    with pytest.raises(ValidationError, match="unknown constraint 'capacty'"):
        make(constraints={"capacty": {}})
    with pytest.raises(ValidationError) as exc:
        make(constraints={"route_duration": {"max": -3}, "time_windows": {"waiting": "maybe"}})
    locs = {loc[:3] for loc, _ in errors_of(exc)}
    assert ("constraints", "route_duration", "max") in locs
    assert ("constraints", "time_windows", "waiting") in locs
    with pytest.raises(ValidationError, match="mapping"):
        make(constraints=["capacity"])


def test_reserved_constraint_needs_planned_status():
    with pytest.raises(ValidationError, match=r"constraints\.energy.*meta\.status: planned"):
        make(constraints={"energy": {"anything": 1}})
    assert make(
        constraints={"energy": {"anything": 1}}, meta={"status": "planned"}
    ).reserved_features() == ("constraints.energy",)


# ------------------------------------------------------------------ dynamics


def test_dynamics_spec():
    spec = make(
        dynamics={
            "arrivals": {"process": "nhpp", "profile": {"breaks": [0, 10, 20], "rates": [1, 0]}},
            "request_class": {"demand": {"dist": "categorical", "values": [1], "probs": [1]}},
            "cutoff": 15,
            "max_requests": 50,
        }
    )
    assert spec.dynamics is not None
    assert spec.dynamics.arrivals is not None and spec.dynamics.arrivals.name == "nhpp"
    assert spec.dynamics.overflow == "error"


@pytest.mark.parametrize(
    ("dynamics", "match"),
    [
        ({"arrivals": {"process": "poisson", "rate": 1}}, "max_requests"),
        ({"arrivals": {"process": "poisson", "rate": 0}, "max_requests": 5}, "greater than 0"),
        ({"arrivals": {"process": "hawkes"}, "max_requests": 5}, "unknown arrival process"),
        (
            {
                "arrivals": {"process": "nhpp", "profile": {"breaks": [0, 1], "rates": [1, 2]}},
                "max_requests": 5,
            },
            "one more entry",
        ),
        (
            {
                "arrivals": {"process": "nhpp", "profile": {"breaks": [0, 2, 1], "rates": [1, 2]}},
                "max_requests": 5,
            },
            "strictly increasing",
        ),
    ],
)
def test_invalid_dynamics(dynamics, match):
    with pytest.raises(ValidationError, match=match):
        make(dynamics=dynamics)


# ------------------------------------------------------------------ decisions


def test_decisions_rules():
    with pytest.raises(ValidationError, match="unknown interface"):
        make(decisions={"interface": "teleport"})
    with pytest.raises(ValidationError, match="requires `routing`"):
        make(decisions={"interface": "accept_reject"})
    with pytest.raises(ValidationError, match="unknown routing rule"):
        make(decisions={"interface": "assign", "routing": "magic"})
    with pytest.raises(ValidationError, match="`period` is required"):
        make(decisions={"epochs": "periodic"})
    with pytest.raises(ValidationError, match="`period` is required"):
        make(decisions={"period": 5})
    assert make(decisions={"epochs": "periodic", "period": 5}).decisions.period == 5
    with pytest.raises(ValidationError, match="sequential epochs"):
        make(
            decisions={"epochs": "sequential"},
            dynamics={"arrivals": {"process": "poisson", "rate": 1}, "max_requests": 5},
        )


@pytest.mark.parametrize(
    ("decisions", "feature"),
    [
        ({"interface": "plan"}, "decisions.interface"),
        ({"epochs": "hybrid"}, "decisions.epochs"),
        ({"commitment": "diversion"}, "decisions.commitment"),
    ],
)
def test_reserved_decisions(decisions, feature):
    with pytest.raises(ValidationError, match="not implemented"):
        make(decisions=decisions)
    assert feature in make(decisions=decisions, meta={"status": "planned"}).reserved_features()


# ------------------------------------------------------------------ recourse


def test_recourse_shorthand_and_full_form():
    spec = make(recourse={"capacity_failure": "return_to_depot"})
    assert spec.recourse.rules["capacity_failure"].name == "return_to_depot"
    assert spec.to_dict()["recourse"] == {
        "capacity_failure": {
            "rule": "return_to_depot",
            "partial_service": True,
            "on_exact_depletion": "continue",
        }
    }
    rule = RecourseRuleSpec.model_validate("split")
    assert rule.model_dump() == "split"  # no parameters: the short form


def test_recourse_rule_parameters():
    rule = RecourseRuleSpec.model_validate({"rule": "outsource", "per_unit": 2})
    assert rule.model_dump() == {
        "rule": "outsource",
        "per_customer": 0.0,
        "per_unit": 2.0,
        "partial_service": True,
    }
    with pytest.raises(ValidationError, match="outsourcing needs its cost"):
        RecourseRuleSpec.model_validate("outsource")
    with pytest.raises(ValidationError, match="outsourcing needs its cost"):
        RecourseRuleSpec.model_validate("agent_decides")
    only_restock = RecourseRuleSpec.model_validate(
        {"rule": "agent_decides", "options": ["return_to_depot"]}
    )
    assert only_restock.params.options == ("return_to_depot",)  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValidationError, match="twice"):
        RecourseRuleSpec.model_validate(
            {"rule": "agent_decides", "options": ["outsource", "outsource"], "per_unit": 1}
        )
    with pytest.raises(ValidationError, match="on_exact_depletion"):
        RecourseRuleSpec.model_validate({"rule": "return_to_depot", "on_exact_depletion": "maybe"})
    with pytest.raises(ValidationError, match=r"capacity_failure\.end_early"):
        make(
            recourse={"capacity_failure": {"rule": "split", "end_early": False}}
        )  # now fleet.end_route


def test_recourse_errors():
    with pytest.raises(ValidationError, match="unknown failure type"):
        make(recourse={"flat_tire": "return_to_depot"})
    with pytest.raises(ValidationError, match="unknown recourse rule"):
        make(recourse={"capacity_failure": "teleport"})
    with pytest.raises(ValidationError, match="mapping"):
        make(recourse=["return_to_depot"])
    with pytest.raises(ValidationError, match=r"recourse\.time_window_failure"):
        make(recourse={"time_window_failure": "return_to_depot"})


# ------------------------------------------------------------------ objective


def test_objective_forms():
    weighted = ObjectiveSpec.model_validate({"distance": 1, "rejected": 50})
    assert weighted.components() == ("distance", "rejected")
    assert weighted.model_dump() == {"distance": 1.0, "rejected": 50.0}
    lex = ObjectiveSpec.model_validate({"lexicographic": ["vehicles", "distance"]})
    assert lex.components() == ("vehicles", "distance")
    assert lex.model_dump() == {"lexicographic": ["vehicles", "distance"]}
    assert ObjectiveSpec.model_validate(lex.model_dump()) == lex


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({}, "at least one"),
        ({"distance": 1, "lexicographic": ["distance"]}, "not both"),
        ({"lexicographic": ["distance", "distance"]}, "unique"),
        ({"bogus": 1}, "unknown cost component"),
        ({"distance": -1}, "only served_demand may be negative"),
        ({"distance": float("inf")}, "finite"),
        ({"lexicographic": ["served_demand", "distance"]}, "revenue components"),
    ],
)
def test_invalid_objectives(raw, match):
    with pytest.raises(ValidationError, match=match):
        ObjectiveSpec.model_validate(raw)


def test_revenue_components_take_negative_weights():
    objective = ObjectiveSpec.model_validate({"served_demand": -1.0, "distance": 0.5})
    assert objective.scalar_weights() == {"served_demand": -1.0, "distance": 0.5}


def test_reserved_cost_component():
    with pytest.raises(ValidationError, match=r"objective\.energy"):
        make(objective={"energy": 1.0})


# ------------------------------------------------------------------ agents, network, ev


def test_agents_rules():
    assert (
        make(agents={"mode": "multi", "reward": "mixed", "reward_mix": 0.5}).agents.reward_mix
        == 0.5
    )
    with pytest.raises(ValidationError, match="reward_mix"):
        make(agents={"reward": "mixed"})
    with pytest.raises(ValidationError, match="reward_mix"):
        make(agents={"reward_mix": 0.3})


def test_network_variants_and_reserved_time_model():
    assert make(network={"type": "matrix"}).network.name == "matrix"
    with pytest.raises(ValidationError, match="unknown network type"):
        make(network={"type": "osm"})
    with pytest.raises(ValidationError):
        make(network={"type": "euclidean", "rounding": "truncate"})
    with pytest.raises(ValidationError, match=r"network\.time_model"):
        make(network={"type": "euclidean", "time_model": {"type": "speed_profile"}})


def test_ev_section_is_reserved_and_opaque():
    with pytest.raises(ValidationError, match="ev"):
        make(ev={"battery": {"capacity": 1}})
    spec = make(ev={"battery": {"capacity": "from_instance"}}, meta={"status": "planned"})
    assert spec.ev["battery"]["capacity"] == "from_instance"
    assert spec.to_dict()["ev"] == {"battery": {"capacity": "from_instance"}}


def test_instance_source_variants():
    with pytest.raises(ValidationError, match="does not match any of the expected tags"):
        make(instance_source={"type": "generator", "name": "uniform"})  # no generators (ADR 0023)
    files = make(instance_source={"type": "files", "paths": ["a.vrp"], "format": "vrplib"})
    assert files.instance_source is not None and files.instance_source.type == "files"
    with pytest.raises(ValidationError):
        make(instance_source={"type": "files", "paths": []})


# ------------------------------------------------------------------ hashing


def test_hash_ignores_meta_and_source_and_includes_defaults():
    a = make()
    b = make(meta={"name": "Another name", "references": ["x"]})
    assert a.content_hash() == b.content_hash()
    explicit = make(decisions={"interface": "next_node", "epochs": "on_event"})
    assert explicit.content_hash() == a.content_hash()
    assert make(decisions={"epochs": "sequential"}).content_hash() != a.content_hash()
    assert a.with_source("somewhere.yaml").content_hash() == b.content_hash()
    assert a.source == "somewhere.yaml"
    assert len(a.short_hash()) == 12


def test_hash_is_insensitive_to_input_types_and_order():
    a = ProblemSpec.model_validate(spec_dict(fleet={"types": [{"name": "t", "capacity": 100}]}))
    b = ProblemSpec.model_validate(
        {
            "objective": {"distance": 1},
            "constraints": {"capacity": {}},
            "fleet": {"types": [{"capacity": 100.0, "name": "t"}]},
        }
    )
    assert a.content_hash() == b.content_hash()


@pytest.mark.parametrize(
    ("grammar", "rule"),
    [
        ({"remainder": "come_back"}, "return_to_depot"),
        (
            {"serve": "none", "remainder": "come_back"},
            {"rule": "return_to_depot", "partial_service": False},
        ),
        (
            {"remainder": "come_back", "on_exact_depletion": "restock"},
            {"rule": "return_to_depot", "on_exact_depletion": "restock"},
        ),
        ({"remainder": "open"}, "split"),
        ({"remainder": "outsource", "per_unit": 2}, {"rule": "outsource", "per_unit": 2}),
        (
            {"serve": "none", "remainder": "outsource", "per_unit": 1},
            {"rule": "outsource", "per_unit": 1, "partial_service": False},
        ),
        (
            {"remainder": "ask", "options": ["return_to_depot"]},
            {"rule": "agent_decides", "options": ["return_to_depot"]},
        ),
    ],
)
def test_recourse_grammar_is_the_same_rule(grammar, rule):
    """`{serve, remainder}` is the documented form; it stores (and hashes) as the rule itself."""
    a = make(recourse={"capacity_failure": grammar})
    b = make(recourse={"capacity_failure": rule})
    assert a.content_hash() == b.content_hash()
    assert a.to_dict()["recourse"] == b.to_dict()["recourse"]


@pytest.mark.parametrize(
    ("grammar", "match"),
    [
        ({"serve": "half", "remainder": "come_back"}, "what_fits or none"),
        ({"serve": "none", "remainder": "open"}, "needs `serve: what_fits`"),
        ({"rule": "split", "remainder": "open"}, "not both"),
        ({"remainder": "teleport"}, "teleport"),
    ],
)
def test_recourse_grammar_errors(grammar, match):
    with pytest.raises(ValidationError, match=match):
        make(recourse={"capacity_failure": grammar})
