import pytest

from fleetgym.errors import ConfigError, NotInScopeError
from fleetgym.spec import REGISTRY, load_spec
from fleetgym.spec.models import InstanceRef

ALL_IDS = REGISTRY.ids(include_planned=True)


@pytest.mark.parametrize("env_id", ALL_IDS)
def test_every_shipped_config_validates(env_id: str):
    spec = load_spec(env_id)
    assert spec.id == env_id
    if spec.meta.status != "planned":
        assert spec.reserved_features() == ()


def test_cvrp_matches_design_example():
    spec = load_spec("CVRP-v0")
    assert spec.network.params.model_dump()["rounding"] == "nearest_int"
    assert spec.fleet.sizing == "unlimited"
    assert spec.fleet.types[0].capacity == InstanceRef()
    assert spec.decisions.epochs == "sequential"
    assert spec.objective.model_dump() == {"distance": 1.0}


def test_vrptw_inherits_and_overrides_cvrp():
    spec = load_spec("VRPTW-v0")
    assert spec.network.params.model_dump()["rounding"] == "none"
    assert spec.constraints.names() == ("capacity", "time_windows", "route_duration")
    assert spec.objective.lexicographic == ("vehicles", "distance")
    assert spec.meta.references == ("solomon1987",)


def test_dvrp_replaces_the_lexicographic_objective():
    spec = load_spec("DVRP-StochReq-v0")
    assert spec.objective.lexicographic is None
    assert spec.objective.model_dump() == {"distance": 1.0, "rejected": 50.0}
    assert spec.decisions.interface == "accept_reject"
    assert spec.dynamics is not None and spec.dynamics.max_requests == 250


def test_evrp_is_planned_with_reserved_features():
    spec = load_spec("EVRP-NL-v0")
    assert spec.meta.status == "planned"
    assert set(spec.reserved_features()) == {"ev", "constraints.energy"}


def test_reserved_features_need_planned_status():
    with pytest.raises(ConfigError, match=r"meta\.status: planned"):
        load_spec("EVRP-NL-v0", overrides=["meta.status=experimental"])


def test_not_in_scope_error_lists_features():
    err = NotInScopeError("EV routing", features=("ev", "constraints.energy"))
    assert str(err) == "EV routing (reserved features: ev, constraints.energy)"
