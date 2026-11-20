from pathlib import Path

import pytest

from fleetgym.errors import ConfigError
from fleetgym.spec import ConfigRegistry, ProblemSpec, load_spec, resolve, resolve_raw
from tests.spec.helpers import BASE, spec_dict, write_yaml


@pytest.fixture
def registry(tmp_path: Path) -> ConfigRegistry:
    root = tmp_path / "configs"
    write_yaml(root / "fam" / "Base-v0.yaml", {"id": "Base-v0", "meta": {"name": "Base"}, **BASE})
    write_yaml(
        root / "fam" / "Child-v0.yaml",
        {
            "id": "Child-v0",
            "extends": "Base-v0",
            "constraints": {"time_windows": {"soft": True}},
            "objective": {"lexicographic": ["vehicles", "distance"]},
        },
    )
    return ConfigRegistry([(root, "test")])


def test_load_by_id_and_extends_chain(registry: ConfigRegistry, tmp_path: Path):
    grandchild = write_yaml(
        tmp_path / "user" / "gc.yaml",
        {"extends": "Child-v0", "constraints": {"capacity": None}, "objective": {"distance": 2}},
    )
    spec = load_spec(grandchild, registry=registry)
    assert spec.id is None  # id is not inherited
    assert spec.meta.name is None  # meta is not inherited
    assert spec.constraints.names() == ("time_windows",)
    assert spec.objective.model_dump() == {"distance": 2.0}  # replaced, not merged
    assert spec.source == str(grandchild.resolve())
    child = load_spec("Child-v0", registry=registry)
    assert child.id == "Child-v0"
    assert child.constraints.names() == ("capacity", "time_windows")


def test_extends_by_relative_path(tmp_path: Path):
    write_yaml(tmp_path / "a" / "parent.yaml", BASE)
    child = write_yaml(
        tmp_path / "b" / "child.yaml",
        {"extends": "../a/parent.yaml", "decisions": {"epochs": "sequential"}},
    )
    spec = load_spec(child, registry=ConfigRegistry([]))
    assert spec.decisions.epochs == "sequential"


def test_extends_cycle_is_reported(tmp_path: Path):
    write_yaml(tmp_path / "a.yaml", {"extends": "b.yaml", **BASE})
    write_yaml(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ConfigError, match="cycle"):
        load_spec(tmp_path / "a.yaml", registry=ConfigRegistry([]))


def test_extends_unknown_id_names_the_child(tmp_path: Path, registry: ConfigRegistry):
    child = write_yaml(tmp_path / "c.yaml", {"extends": "Bse-v0"})
    with pytest.raises(
        ConfigError,
        match=r"c\.yaml: extends: unknown environment ID 'Bse-v0'; did you mean Base-v0",
    ):
        load_spec(child, registry=registry)


def test_extends_must_be_a_string(tmp_path: Path):
    child = write_yaml(tmp_path / "c.yaml", {"extends": ["a"]})
    with pytest.raises(ConfigError, match="extends"):
        load_spec(child, registry=ConfigRegistry([]))


def test_errors_name_the_file_that_set_the_value(registry: ConfigRegistry, tmp_path: Path):
    child = write_yaml(
        tmp_path / "c.yaml", {"extends": "Base-v0", "constraints": {"route_duration": {"max": -1}}}
    )
    with pytest.raises(ConfigError) as exc:
        load_spec(child, registry=registry)
    assert exc.value.source == str(child.resolve())
    assert exc.value.key_path == "constraints.route_duration.max"


def test_errors_in_inherited_values_point_to_the_parent(tmp_path: Path):
    parent = write_yaml(
        tmp_path / "p.yaml", spec_dict(fleet={"types": [{"name": "t", "capacity": "lots"}]})
    )
    child = write_yaml(
        tmp_path / "c.yaml", {"extends": "p.yaml", "decisions": {"epochs": "sequential"}}
    )
    with pytest.raises(ConfigError) as exc:
        load_spec(child, registry=ConfigRegistry([]))
    assert exc.value.source == str(parent.resolve())
    assert exc.value.key_path == "fleet.types.0.capacity"


def test_all_errors_are_reported_at_once(tmp_path: Path):
    path = write_yaml(
        tmp_path / "x.yaml",
        spec_dict(decisions={"interface": "nope"}, agents={"mode": "many"}, objective={"bogus": 1}),
    )
    with pytest.raises(ConfigError) as exc:
        load_spec(path, registry=ConfigRegistry([]))
    keys = {exc.value.key_path} | {k for _, k, _ in exc.value.details}
    assert {"decisions", "agents.mode", "objective"} <= keys
    assert str(exc.value).count("\n  - ") == len(exc.value.details)


def test_missing_required_section_is_reported(tmp_path: Path):
    path = write_yaml(tmp_path / "x.yaml", {"objective": {"distance": 1}})
    with pytest.raises(ConfigError, match=r"x\.yaml: fleet: Field required"):
        load_spec(path, registry=ConfigRegistry([]))


def test_overrides_apply_after_extends_and_are_named_in_errors(registry: ConfigRegistry):
    spec = load_spec(
        "Child-v0", overrides=["constraints.time_windows.soft=false"], registry=registry
    )
    assert spec.constraints["time_windows"].model_dump()["soft"] is False
    with pytest.raises(ConfigError) as exc:
        load_spec("Child-v0", overrides=["fleet.types.0.capacity=-5"], registry=registry)
    assert exc.value.source == "<override fleet.types.0.capacity=-5>"
    assert exc.value.key_path == "fleet.types.0.capacity"


def test_load_from_mapping_and_from_spec():
    spec = load_spec(spec_dict())
    assert spec.source == "<dict>"
    assert load_spec(spec) is spec
    changed = load_spec(spec, overrides=["decisions.epochs=sequential"])
    assert changed.decisions.epochs == "sequential"


def test_round_trip_through_yaml_keeps_the_hash(registry: ConfigRegistry, tmp_path: Path):
    spec = load_spec("Child-v0", registry=registry)
    path = tmp_path / "dump.yaml"
    path.write_text(spec.to_yaml())
    again = load_spec(path, registry=registry)
    assert again.content_hash() == spec.content_hash()
    assert again == spec


def test_hash_ignores_yaml_formatting(tmp_path: Path):
    a = tmp_path / "a.yaml"
    a.write_text(
        "fleet: {types: [{name: t, capacity: 100}]}\n"
        "constraints: {capacity: {}}\n"
        "objective: {distance: 1}\n"
    )
    b = tmp_path / "b.yaml"
    b.write_text(
        "# comment\nobjective:\n  distance: 1.0\nconstraints:\n  capacity: {}\n"
        "fleet:\n  types:\n    - capacity: 1.0e2\n      name: t\n"
    )
    reg = ConfigRegistry([])
    assert load_spec(a, registry=reg).content_hash() == load_spec(b, registry=reg).content_hash()


def test_resolve_raw_orders_sections_and_records_layers(registry: ConfigRegistry):
    raw = resolve_raw("Child-v0", registry=registry)
    assert list(raw)[:2] == ["id", "fleet"]
    resolved = resolve("Child-v0", overrides=["agents.mode=multi"], registry=registry)
    assert [s if s.startswith("<") else Path(s).name for s, _ in resolved.layers] == [
        "Base-v0.yaml",
        "Child-v0.yaml",
        "<override agents.mode=multi>",
    ]
    assert resolved.source.endswith("Child-v0.yaml")
    assert resolved.locate(("constraints", "capacity")).endswith("Base-v0.yaml")
    assert resolved.locate(("agents", "mode")) == "<override agents.mode=multi>"


def test_problem_spec_is_exported():
    assert isinstance(load_spec(spec_dict()), ProblemSpec)
