"""Registered configs as pytest parameters.

Configs whose instance comes from a benchmark collection download it on first use, so they carry the
``network`` marker (never in the default run); tests of their dynamics use local instances instead.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

import fleetgym as fg
from fleetgym.spec import load_spec
from fleetgym.spec.models import CollectionSource


def needs_download(env_id: str) -> bool:
    """True if the config's instance source is a benchmark collection."""
    return isinstance(load_spec(env_id).instance_source, CollectionSource)


def registered(agents: Literal["single", "multi"] | None = None) -> list[Any]:
    """Registered, runnable IDs as parameters; downloading ones are marked ``network``."""
    return [
        pytest.param(
            env_id, id=env_id, marks=[pytest.mark.network] if needs_download(env_id) else []
        )
        for env_id in fg.REGISTRY.ids(agents=agents)
    ]
