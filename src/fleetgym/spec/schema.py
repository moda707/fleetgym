"""JSON Schema of config files, for editor autocompletion and inline errors.

The schema describes one YAML file *before* ``extends`` is applied, so it cannot require fields
that a parent may provide: required lists are dropped except plugin discriminators (``dist``,
``type``, ...). Completeness is checked by `fleetgym.spec.load_spec` and ``fleetgym validate``.

A copy ships next to the configs (``fleetgym/configs/fleetgym.schema.json``); a test keeps it in
sync with the models.
"""

from __future__ import annotations

from typing import Any

from fleetgym.spec.merge import DISCRIMINATORS
from fleetgym.spec.models import ProblemSpec
from fleetgym.spec.registries import UNCERTAINTY_SOURCES

__all__ = ["SCHEMA_FILENAME", "json_schema"]

SCHEMA_FILENAME = "fleetgym.schema.json"
_DRAFT = "https://json-schema.org/draft/2020-12/schema"


def _relax(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: _relax(v) for k, v in node.items()}
        if "required" in out:
            kept = [r for r in out["required"] if r in DISCRIMINATORS]
            if kept:
                out["required"] = kept
            else:
                del out["required"]
        if "oneOf" in out:
            out["anyOf"] = out.pop("oneOf")
        return out
    if isinstance(node, list):
        return [_relax(v) for v in node]
    return node


def json_schema() -> dict[str, Any]:
    """Return the JSON Schema of a FleetGym config file."""
    schema = _relax(ProblemSpec.model_json_schema(mode="validation"))
    schema["properties"]["uncertainty"] = {
        "type": "object",
        "properties": {
            name: {"$ref": "#/$defs/UncertainSpec", "description": entry.description}
            for name, entry in UNCERTAINTY_SOURCES.items()
        },
        "additionalProperties": False,
    }
    schema["$defs"].pop("FrozenMap_str_UncertainSpec_", None)
    schema["properties"]["extends"] = {
        "type": "string",
        "description": "Environment ID or relative path of the config this one extends.",
    }
    return {
        "$schema": _DRAFT,
        **schema,
        "title": "FleetGym config",
        "description": "One FleetGym environment. Values may come from the instance with "
        "`from_instance`; `null` removes an inherited value.",
    }
