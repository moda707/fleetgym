"""Shared builders for spec tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

BASE: dict[str, Any] = {
    "fleet": {"types": [{"name": "truck", "capacity": 100}]},
    "constraints": {"capacity": {}},
    "objective": {"distance": 1.0},
}


def spec_dict(**changes: Any) -> dict[str, Any]:
    """Return a minimal valid spec dict with top-level sections replaced by ``changes``."""
    data = copy.deepcopy(BASE)
    data.update(copy.deepcopy(changes))
    return data


def write_yaml(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
