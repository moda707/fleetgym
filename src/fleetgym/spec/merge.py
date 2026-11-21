"""Merge rules for ``extends`` (implementation plan §3.3, ADR 0006).

* Maps merge deeply; scalars and lists in the child replace the parent's.
* ``constraints`` merges by name (it is a map keyed by constraint name).
* ``null`` in the child deletes the inherited key, so the default applies.
* If a child map changes a discriminator (``dist``, ``type``, ``process``, ``model``,
  ``function``, ``rule``), the subtree is replaced instead of merged. A recourse rule written as
  ``rule`` in one and ``remainder`` in the other (two spellings of one choice) is replaced too.
* ``objective`` is always replaced, never merged.
* ``meta`` and ``id`` are never inherited.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "DISCRIMINATORS",
    "NOT_INHERITED",
    "REPLACED_SECTIONS",
    "drop_nulls",
    "merge_configs",
    "merge_maps",
]

DISCRIMINATORS = ("dist", "type", "process", "model", "function", "rule")
REPLACED_SECTIONS = frozenset({"objective"})
NOT_INHERITED = frozenset({"meta", "id", "extends"})


def drop_nulls(value: Any) -> Any:
    """Remove ``None`` values from mappings, recursively (lists are kept as they are)."""
    if isinstance(value, Mapping):
        return {k: drop_nulls(v) for k, v in value.items() if v is not None}
    return value


#: Keys that name the same choice in two spellings (a recourse rule: ``rule`` or ``remainder``).
EQUIVALENT_DISCRIMINATORS = (("rule", "remainder"),)


def _changes_discriminator(parent: Mapping[str, Any], child: Mapping[str, Any]) -> bool:
    if any(d in parent and d in child and parent[d] != child[d] for d in DISCRIMINATORS):
        return True
    for group in EQUIVALENT_DISCRIMINATORS:
        parent_keys = {k for k in group if k in parent}
        child_keys = {k for k in group if k in child}
        if parent_keys and child_keys and parent_keys != child_keys:
            return True
    return any(
        k in parent and k in child and parent[k] != child[k]
        for group in EQUIVALENT_DISCRIMINATORS
        for k in group
    )


def merge_maps(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``child`` into ``parent`` with the rules of this module (no top-level rules)."""
    out = dict(parent)
    for key, value in child.items():
        if value is None:
            out.pop(key, None)
            continue
        inherited = out.get(key)
        if (
            isinstance(value, Mapping)
            and isinstance(inherited, Mapping)
            and not _changes_discriminator(inherited, value)
        ):
            out[key] = merge_maps(inherited, value)
        else:
            out[key] = drop_nulls(value)
    return out


def merge_configs(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """Merge a child config into its resolved parent.

    Parameters
    ----------
    parent
        The parent config, already resolved (its own ``extends`` applied).
    child
        The child config as written, without its ``extends`` key.
    """
    base = {k: v for k, v in parent.items() if k not in NOT_INHERITED}
    for section in REPLACED_SECTIONS & child.keys():
        base.pop(section, None)
    return merge_maps(base, child)
