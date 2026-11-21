"""Problem specification (layer L2).

Frozen pydantic models, YAML loading, merging and the config registry.

Examples
--------
>>> from fleetgym.spec import load_spec
>>> spec = load_spec("CVRP-v1")
>>> spec.decisions.interface
'next_node'
"""

from fleetgym.spec.loader import load_spec, resolve, resolve_raw
from fleetgym.spec.models import ProblemSpec
from fleetgym.spec.registry import REGISTRY, ConfigEntry, ConfigRegistry
from fleetgym.spec.schema import json_schema

__all__ = [
    "REGISTRY",
    "ConfigEntry",
    "ConfigRegistry",
    "ProblemSpec",
    "json_schema",
    "load_spec",
    "resolve",
    "resolve_raw",
]
