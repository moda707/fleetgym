"""Instance schemas, the FleetGym JSON format and instance sets (ADR 0023)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import fleetgym as fg
from fleetgym.data import (
    CORE_SCHEMA,
    Instance,
    get_schema,
    load_schema,
    read_instance,
    read_instance_set,
    register_schema,
    write_instance,
    write_instance_set,
)
from fleetgym.data.schema import build_schema
from fleetgym.errors import InstanceError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "instances"

RURAL = """\
schema: tests/rural/1
extends: fleetgym/core/1
node:
  village: {type: int, min: 0, description: Village of the customer}
global:
  courier_price: {min: 0, unit: money}
vehicle_type:
  electric: {type: bool}
"""


def rural_schema(tmp_path: Path):
    path = tmp_path / "rural.schema.yaml"
    path.write_text(RURAL, encoding="utf-8")
    return load_schema(path)


def small(**changes) -> Instance:
    fields = {
        "name": "small",
        "source": "test",
        "demands": np.array([0.0, 2.0, 3.0]),
        "coords": np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]),
        "vehicle_types": ({"capacity": 5.0, "count": 2},),
    }
    fields.update(changes)
    return Instance(**fields)


# ------------------------------------------------------------------ schemas


def test_core_schema_declares_the_built_in_fields():
    assert CORE_SCHEMA.id == "fleetgym/core/1"
    assert {"kind", "x", "y", "demand", "deadline"} <= set(CORE_SCHEMA.names("node"))
    assert CORE_SCHEMA.field("node", "demand").required  # pyright: ignore[reportOptionalMemberAccess]
    assert CORE_SCHEMA.level_of("max_route_duration") == "global"
    assert CORE_SCHEMA.level_of("capacity") == "vehicle_type"
    assert get_schema("fleetgym/core/1") is CORE_SCHEMA


def test_user_schemas_extend_and_inherit(tmp_path):
    rural = rural_schema(tmp_path)
    assert rural.extends == "fleetgym/core/1"
    assert rural.field("node", "demand") == CORE_SCHEMA.field("node", "demand")
    assert rural.field("node", "village").type == "int"  # pyright: ignore[reportOptionalMemberAccess]
    assert load_schema(tmp_path / "rural.schema.yaml") is rural  # identical: same registration
    assert rural.content_hash() != CORE_SCHEMA.content_hash()
    assert rural.to_dict()["global"]["courier_price"]["unit"] == "money"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"node": {}}, "needs `schema"),
        ({"schema": "only/two"}, "<namespace>/<name>/<version>"),
        ({"schema": "t/a/1", "extends": "nobody/x/1"}, "unknown instance schema"),
        ({"schema": "t/b/1", "nodes": {}}, "unknown keys"),
        (
            {"schema": "t/c/1", "extends": "fleetgym/core/1", "node": {"demand": {}}},
            "already declared",
        ),
        ({"schema": "t/d/1", "global": {"_secret": {}}}, "reserved"),
        ({"schema": "t/e/1", "global": {"x1": {"type": "complex"}}}, "is not float"),
        ({"schema": "t/f/1", "global": {"x1": {"unit": "parsecs"}}}, "unknown unit"),
        ({"schema": "t/g/1", "global": {"x1": {"min": "low"}}}, "finite number"),
        ({"schema": "t/h/1", "global": {"x1": {"colour": "red"}}}, "unknown properties"),
    ],
)
def test_malformed_schemas_are_rejected(raw, match):
    with pytest.raises(InstanceError, match=match):
        build_schema(raw)


def test_a_schema_id_cannot_be_reused_with_other_fields():
    register_schema({"schema": "tests/once/1", "global": {"a": {}}})
    with pytest.raises(InstanceError, match="already registered"):
        register_schema({"schema": "tests/once/1", "global": {"b": {}}})


def test_required_fields_choices_and_bounds(tmp_path):
    register_schema(
        {
            "schema": "tests/strict/1",
            "extends": "fleetgym/core/1",
            "global": {
                "budget": {"required": True, "min": 0},
                "mode": {"type": "str", "choices": ["a", "b"]},
            },
            "vehicle_type": {"licence": {"type": "str", "required": True}},
        }
    )
    with pytest.raises(InstanceError, match="required global field 'budget'"):
        small(schema="tests/strict/1", vehicle_types=({"capacity": 5, "licence": "B"},))
    with pytest.raises(InstanceError, match="required vehicle type field 'licence'"):
        small(schema="tests/strict/1", extra={"budget": 1})
    with pytest.raises(InstanceError, match="'c' is not one of a, b"):
        small(
            schema="tests/strict/1",
            extra={"budget": 1, "mode": "c"},
            vehicle_types=({"licence": "B"},),
        )
    ok = small(
        schema="tests/strict/1", extra={"budget": 1, "mode": "a"}, vehicle_types=({"licence": "B"},)
    )
    assert ok.schema_problems() == []


# ------------------------------------------------------------------ the JSON format


def test_json_round_trip_keeps_every_field_and_the_hash(tmp_path):
    rural_schema(tmp_path)
    instance = small(
        schema="tests/rural/1",
        time_windows=np.array([[0, 100], [5, 50], [10, 60]]),
        service_times=np.array([0.0, 2.0, 3.0]),
        extra={"village": np.array([0, 1, 1]), "courier_price": 2.5, "max_route_duration": 90.0},
        vehicle_types=({"capacity": 5.0, "count": 2, "electric": True},),
        meta={"comment": "hand-made"},
    )
    path = write_instance(instance, tmp_path / "small.json")
    document = json.loads(path.read_text())
    assert document["nodes"]["kind"] == ["depot", "customer", "customer"]
    assert document["global"]["courier_price"] == 2.5
    back = read_instance(path)
    assert back.content_hash() == instance.content_hash()
    assert back.meta["comment"] == "hand-made"
    np.testing.assert_array_equal(back.extra["village"], [0, 1, 1])


@pytest.mark.parametrize(
    ("file", "fmt"),
    [("tiny.vrp", None), ("tiny_solomon.txt", "solomon"), ("tiny_explicit.vrp", None)],
)
def test_benchmark_files_convert_to_the_format(tmp_path, file, fmt):
    original = read_instance(FIXTURES / file, fmt=fmt)
    back = read_instance(write_instance(original, tmp_path / "x.json"))
    assert back.content_hash() == original.content_hash()


def test_json_documents_name_the_file_and_field(tmp_path):
    def write(document) -> Path:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(document))
        return path

    base = {
        "format": "fleetgym-instance/1",
        "nodes": {"kind": ["depot", "customer"], "demand": [0, 1]},
    }
    assert read_instance(write(base)).n_customers == 1
    cases = [
        ({**base, "format": "csv"}, "not a FleetGym instance"),
        ({**base, "extras": {}}, "unknown top-level keys"),
        ({**base, "nodes": {"kind": ["customer", "depot"], "demand": [1, 0]}}, "depots first"),
        ({**base, "nodes": {**base["nodes"], "x": [0, 1]}}, "go together"),
        (
            {**base, "nodes": {**base["nodes"], "colour": [1, 2]}},
            "nodes.colour is not a node field",
        ),
        ({**base, "global": {"budget": 1}}, "global.budget is not a global field"),
        ({**base, "nodes": {**base["nodes"], "demand": [0]}}, "list of 2 values"),
        ({**base, "nodes": {**base["nodes"], "demand": [0, -1]}}, "non-negative"),
        ({**base, "schema": "nobody/x/1"}, "unknown instance schema"),
    ]
    for document, match in cases:
        with pytest.raises(InstanceError, match=match) as exc:
            read_instance(write(document))
        assert "bad.json" in str(exc.value)


def test_json_can_point_to_its_schema_file(tmp_path):
    (tmp_path / "rural2.schema.yaml").write_text(RURAL.replace("tests/rural/1", "tests/rural/2"))
    document = {
        "format": "fleetgym-instance/1",
        "schema": "tests/rural/2",
        "schema_file": "rural2.schema.yaml",
        "nodes": {"kind": ["depot", "customer"], "demand": [0, 1], "village": [0, 3]},
    }
    (tmp_path / "v.json").write_text(json.dumps(document))
    assert read_instance(tmp_path / "v.json").extra["village"].tolist() == [0, 3]


# ------------------------------------------------------------------ instance sets


def test_instance_sets_record_checksums_and_detect_changes(tmp_path):
    instances = [small(demands=np.array([0.0, d, 1.0])) for d in (1.0, 2.0, 4.0)]
    made = write_instance_set(instances, tmp_path / "set", name="three", reference="tests")
    assert len(made) == 3 and made.max_nodes == 3 and made.info["reference"] == "tests"
    again = read_instance_set(tmp_path / "set")
    assert again.content_hash() == made.content_hash()
    assert [i.demands[1] for i in again] == [1.0, 2.0, 4.0]
    with pytest.raises(InstanceError, match="already holds"):
        write_instance_set(instances, tmp_path / "set", name="again")
    (tmp_path / "set" / "0002.json").write_text((tmp_path / "set" / "0002.json").read_text() + " ")
    with pytest.raises(InstanceError, match="does not match its checksum"):
        read_instance_set(tmp_path / "set")
    register_schema({"schema": "tests/other/1", "extends": "fleetgym/core/1"})
    mixed = [small(), small(schema="tests/other/1")]
    with pytest.raises(InstanceError, match="follows 'tests/other/1', the set follows"):
        write_instance_set(mixed, tmp_path / "mixed", name="m")


def test_environments_draw_episodes_from_an_instance_set(tmp_path):
    instances = [
        small(
            demands=np.array([0.0] + [1.0] * n),
            coords=np.random.default_rng(n).uniform(0, 10, (n + 1, 2)),
        )
        for n in (3, 5, 4)
    ]
    write_instance_set(instances, tmp_path / "train", name="train")
    spec = {
        "extends": "CVRP-v1",
        "instance_source": {"type": "files", "paths": [str(tmp_path / "train")]},
    }
    env = fg.make(spec)
    assert env.max_nodes == 6
    sizes = set()
    for seed in range(12):
        env.reset(seed=seed)
        sizes.add(env.base_instance.n_customers)
    assert sizes == {3, 4, 5}
