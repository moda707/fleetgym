import json
from importlib import resources
from pathlib import Path

import jsonschema
import pytest
import yaml

from fleetgym.cli.schema import render_schema
from fleetgym.spec import json_schema
from fleetgym.spec.loader import resolve_raw
from fleetgym.spec.registries import (
    CONSTRAINTS,
    COST_COMPONENTS,
    DISTRIBUTIONS,
    UNCERTAINTY_SOURCES,
)
from fleetgym.spec.schema import SCHEMA_FILENAME

SHIPPED = Path(str(resources.files("fleetgym.configs"))) / SCHEMA_FILENAME


def test_shipped_schema_is_up_to_date():
    assert SHIPPED.read_text(encoding="utf-8") == render_schema(), (
        "regenerate with: uv run fleetgym schema -o src/fleetgym/configs/fleetgym.schema.json"
    )


def test_schema_lists_registered_plugins():
    schema = json_schema()
    defs = schema["$defs"]
    assert set(schema["properties"]["constraints"]["$ref"].split("/")[-1:]) == {"ConstraintsSpec"}
    assert set(defs["ConstraintsSpec"]["properties"]) == set(CONSTRAINTS.names())
    assert set(schema["properties"]["uncertainty"]["properties"]) == set(
        UNCERTAINTY_SOURCES.names()
    )
    dist_names = {v["properties"]["dist"]["const"] for v in defs["DistSpec"]["anyOf"]}
    assert dist_names == set(DISTRIBUTIONS.names())
    assert set(defs["ObjectiveSpec"]["properties"]) == {*COST_COMPONENTS.names(), "lexicographic"}
    assert "extends" in schema["properties"]


def test_schema_only_requires_discriminators():
    def walk(node):
        if isinstance(node, dict):
            if "required" in node:
                assert set(node["required"]) <= {
                    "dist",
                    "type",
                    "process",
                    "model",
                    "function",
                    "rule",
                }
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json_schema())


def test_configs_point_editors_at_the_schema():
    root = SHIPPED.parent
    for path in root.rglob("*.yaml"):
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == f"# yaml-language-server: $schema=../{SCHEMA_FILENAME}", path
    json.loads(SHIPPED.read_text(encoding="utf-8"))


CONFIGS = sorted((SHIPPED.parent).rglob("*.yaml"))


def _validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(json.loads(SHIPPED.read_text(encoding="utf-8")))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_shipped_configs_validate_against_the_schema(path: Path):
    """The editor schema accepts every config as written and after `extends` is resolved."""
    validator = _validator()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert not list(validator.iter_errors(raw))
    assert not list(validator.iter_errors(resolve_raw(path)))


def test_schema_rejects_unknown_keys_and_bad_values():
    validator = _validator()
    raw = yaml.safe_load((SHIPPED.parent / "deterministic" / "CVRP-v1.yaml").read_text())
    assert list(validator.iter_errors({**raw, "objectve": {"distance": 1}}))
    assert list(validator.iter_errors({**raw, "network": {"type": "euclidean", "rounding": "up"}}))
