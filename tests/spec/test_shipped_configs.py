import pytest

from fleetgym.errors import ConfigError, NotInScopeError
from fleetgym.spec import REGISTRY, load_spec
from tests.families import spec_for

ALL_IDS = REGISTRY.ids(include_planned=True)


@pytest.mark.parametrize("env_id", ALL_IDS)
def test_every_shipped_config_validates(env_id: str):
    spec = load_spec(env_id)
    assert spec.id == env_id
    if spec.meta.status != "planned":
        assert spec.reserved_features() == ()


def test_cvrp_declares_structure_and_takes_numbers_from_instances():
    spec = load_spec("CVRP-v1")
    assert spec.network.params.model_dump()["rounding"] == "from_instance"
    assert spec.fleet.sizing == "unlimited"
    assert spec.fleet.from_instance
    assert spec.instance_source is None  # FleetGym ships no instances
    assert spec.decisions.epochs == "sequential"
    assert spec.objective.model_dump() == {"distance": 1.0}


def test_vrptw_inherits_and_overrides_cvrp():
    spec = load_spec("VRPTW-v1")
    assert spec.network.params.model_dump()["rounding"] == "from_instance"
    assert spec.constraints.names() == ("capacity", "time_windows", "route_duration")
    assert spec.objective.lexicographic == ("vehicles", "distance")
    assert spec.meta.references == ("solomon1987",)


def test_dvrp_replaces_the_lexicographic_objective():
    spec = spec_for("DVRP-v1")
    assert spec.objective.lexicographic is None
    assert spec.objective.model_dump() == {"distance": 1.0, "rejected": 50.0}
    assert spec.decisions.interface == "accept_reject"
    assert spec.dynamics is not None and spec.dynamics.max_requests == 250


def test_evrp_is_planned_with_reserved_features():
    spec = load_spec("EVRP-NL-v1")
    assert spec.meta.status == "planned"
    assert set(spec.reserved_features()) == {"ev", "constraints.energy"}


def test_reserved_features_need_planned_status():
    with pytest.raises(ConfigError, match=r"meta\.status: planned"):
        load_spec("EVRP-NL-v1", overrides=["meta.status=experimental"])


def test_not_in_scope_error_lists_features():
    err = NotInScopeError("EV routing", features=("ev", "constraints.energy"))
    assert str(err) == "EV routing (reserved features: ev, constraints.energy)"
