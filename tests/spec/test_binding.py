from pathlib import Path

import numpy as np
import pytest

from fleetgym.data import Instance, read_instance
from fleetgym.errors import ConfigError, NotInScopeError
from fleetgym.network import EuclideanNetwork, MatrixNetwork, build_network
from fleetgym.spec import load_spec
from fleetgym.spec.binding import bind, instance_from_source
from tests.spec.helpers import spec_dict, write_yaml

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "instances"


def tiny() -> Instance:
    return read_instance(FIXTURES / "tiny.vrp")


def test_bind_resolves_default_keys_and_symbols():
    spec = load_spec(
        spec_dict(
            fleet={
                "sizing": "fixed",
                "types": [{"name": "t", "capacity": "from_instance", "count": "from_instance"}],
            },
            constraints={
                "capacity": {},
                "time_windows": {},
                "route_duration": {"max": "depot_due_time"},
            },
            uncertainty={
                "demand": {"dist": "poisson", "mean": "from_instance", "revealed": "on_arrival"}
            },
        )
    )
    inst = read_instance(FIXTURES / "tiny_solomon.txt")
    problem = bind(spec, inst)
    assert problem.vehicle_types[0].capacity == 20.0
    assert problem.vehicle_types[0].count == 3
    assert problem.refs["constraints.route_duration.max"] == 100.0
    np.testing.assert_array_equal(problem.value("uncertainty.demand.mean"), inst.demands)
    assert problem.value("constraints.route_duration.max") == 100.0
    assert problem.value("objective.distance") == 1.0
    assert problem.value("fleet.types.0.name") == "t"
    with pytest.raises(KeyError):
        problem.value("objective.nothing")
    assert problem.spec_hash == spec.content_hash()
    assert problem.instance_hash == inst.content_hash()


def test_explicit_key_and_literal_values():
    spec = load_spec(
        spec_dict(
            fleet={"types": [{"name": "t", "capacity": 7}]},
            constraints={"capacity": {}, "route_duration": {"max": {"from_instance": "dimension"}}},
        )
    )
    problem = bind(spec, tiny())
    assert problem.vehicle_types[0].capacity == 7.0
    assert problem.refs["constraints.route_duration.max"] == 5


def test_missing_instance_value_names_config_path(tmp_path: Path):
    path = write_yaml(
        tmp_path / "c.yaml",
        spec_dict(constraints={"capacity": {}, "route_duration": {"max": "from_instance"}}),
    )
    with pytest.raises(ConfigError) as exc:
        bind(load_spec(path), tiny())
    assert exc.value.key_path == "constraints.route_duration.max"
    assert exc.value.source == str(path.resolve())
    assert "max_route_duration" in exc.value.message


def test_bare_reference_without_default_key_is_an_error():
    spec = load_spec(
        spec_dict(
            dynamics={
                "arrivals": {"process": "poisson", "rate": 1},
                "max_requests": 3,
                "request_class": {"demand": {"dist": "fixed", "value": "from_instance"}},
            }
        )
    )
    with pytest.raises(ConfigError, match=r"no default instance key.*\{from_instance: <key>\}"):
        bind(spec, tiny())


@pytest.mark.parametrize(
    ("fleet", "match"),
    [
        ({"types": [{"name": "t", "capacity": {"from_instance": "demands"}}]}, "positive number"),
        (
            {
                "sizing": "fixed",
                "types": [{"name": "t", "count": {"from_instance": "capacity"}, "capacity": 1}],
            },
            None,
        ),
    ],
)
def test_scalar_checks(fleet, match):
    spec = load_spec(spec_dict(fleet=fleet))
    if match is None:
        assert bind(spec, tiny()).vehicle_types[0].count == 10
    else:
        with pytest.raises(ConfigError, match=match):
            bind(spec, tiny())


def test_non_integer_count_is_rejected():
    inst = Instance(
        name="x",
        source="t",
        demands=np.array([0.0, 1.0]),
        coords=np.array([[0.0, 0.0], [1.0, 1.0]]),
        extra={"k": 2.5},
    )
    spec = load_spec(
        spec_dict(
            fleet={
                "sizing": "fixed",
                "types": [{"name": "t", "capacity": 1, "count": {"from_instance": "k"}}],
            }
        )
    )
    with pytest.raises(ConfigError, match="integer"):
        bind(spec, inst)


@pytest.mark.parametrize(
    ("changes", "instance_file", "match"),
    [
        ({"network": {"type": "matrix"}}, "tiny.vrp", "explicit distance matrix"),
        ({}, "tiny_explicit.vrp", "node coordinates"),
        ({"fleet": {"types": [{"name": "t"}]}}, "tiny.vrp", "needs a capacity"),
        ({"constraints": {"capacity": {}, "time_windows": {}}}, "tiny.vrp", "no time windows"),
    ],
)
def test_compatibility_checks(changes, instance_file, match):
    spec = load_spec(spec_dict(**changes))
    with pytest.raises(ConfigError, match=match):
        bind(spec, read_instance(FIXTURES / instance_file))


def test_multiple_depots_are_rejected():
    inst = Instance(
        name="x",
        source="t",
        demands=np.array([0.0, 0.0, 1.0]),
        coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]]),
        n_depots=2,
    )
    with pytest.raises(ConfigError, match="supports one depot"):
        bind(load_spec(spec_dict()), inst)


def test_planned_specs_cannot_be_bound():
    with pytest.raises(NotInScopeError, match=r"EVRP-NL-v0 is planned.*reserved features: ev"):
        bind(load_spec("EVRP-NL-v0"), tiny())
    with pytest.raises(NotInScopeError, match="VRPSD-Outsource-v0 is planned"):
        bind(load_spec("VRPSD-Outsource-v0"), tiny())


def test_build_network_follows_the_spec():
    euclid = build_network(bind(load_spec("CVRP-v0"), tiny()))
    assert isinstance(euclid, EuclideanNetwork)
    assert euclid.distance(0, 4) == 3.0  # nearest_int: 2.5 rounds up
    plain = build_network(bind(load_spec(spec_dict()), tiny()))
    assert plain.distance(0, 4) == 2.5
    matrix = build_network(
        bind(
            load_spec(spec_dict(network={"type": "matrix"})),
            read_instance(FIXTURES / "tiny_explicit.vrp"),
        )
    )
    assert isinstance(matrix, MatrixNetwork)
    assert matrix.distance(1, 0) == 5


def test_instance_from_generator_source():
    spec = load_spec("VRPTW-v0")
    a = instance_from_source(spec, np.random.default_rng(1))
    b = instance_from_source(spec, np.random.default_rng(1))
    assert a.content_hash() == b.content_hash()
    assert a.time_windows is not None
    problem = bind(spec, a)
    assert problem.refs["constraints.route_duration.max"] == 480.0


def test_instance_from_files_source(tmp_path: Path):
    config = write_yaml(
        tmp_path / "cfg" / "c.yaml",
        spec_dict(
            instance_source={"type": "files", "paths": [str(FIXTURES / "tiny.vrp"), "local.txt"]}
        ),
    )
    (tmp_path / "cfg" / "local.txt").write_text((FIXTURES / "tiny_solomon.txt").read_text())
    spec = load_spec(config)
    names = {instance_from_source(spec, np.random.default_rng(seed)).name for seed in range(20)}
    assert names == {"tiny-n5", "TINY01"}


def test_instance_source_errors(tmp_path: Path):
    with pytest.raises(ConfigError, match="no `instance_source`"):
        instance_from_source(load_spec(spec_dict()), np.random.default_rng(0))
    spec = load_spec(
        spec_dict(instance_source={"type": "files", "paths": [str(tmp_path / "nope.vrp")]})
    )
    with pytest.raises(ConfigError, match="not found"):
        instance_from_source(spec, np.random.default_rng(0))
