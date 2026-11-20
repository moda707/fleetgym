"""Golden snapshots of registered spec hashes.

Every registered config's content hash is stored in ``tests/golden/spec_hashes.json``. A change
to a config's dynamics or defaults changes its hash; the test then fails until either the
environment's version is bumped (``-v0`` → ``-v1``) or the snapshot is regenerated on purpose
with ``uv run pytest tests/spec/test_golden_hashes.py --update-golden``.
"""

from __future__ import annotations

from fleetgym.spec import REGISTRY, ConfigRegistry, load_spec

__all__ = ["registered_spec_hashes"]


def registered_spec_hashes(registry: ConfigRegistry | None = None) -> dict[str, str]:
    """Return ``{env_id: content_hash}`` for every registered config, planned ones included."""
    registry = registry or REGISTRY
    return {
        env_id: load_spec(env_id, registry=registry).content_hash()
        for env_id in registry.ids(include_planned=True)
    }
