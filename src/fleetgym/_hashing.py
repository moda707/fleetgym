"""Canonical JSON serialization and SHA-256 content hashes (layer L0).

A content hash identifies a spec, an instance or a scenario independently of formatting, key
order or the machine it was computed on. The hash scheme is versioned: every hash starts with
``fg1:``. Changing how canonical JSON is produced requires a new prefix.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["HASH_SCHEME", "canonical_json", "content_hash", "short_hash"]

HASH_SCHEME = "fg1"


def canonical_json(obj: Any) -> str:
    """Serialize JSON-compatible data deterministically.

    Keys are sorted, separators carry no whitespace, floats use Python's shortest round-trip
    representation, and NaN or infinity are rejected (they have no portable JSON form).

    Parameters
    ----------
    obj
        Data made of dicts with string keys, lists, strings, numbers, booleans and ``None``.

    Raises
    ------
    ValueError
        If ``obj`` contains NaN or infinity.
    TypeError
        If ``obj`` contains a value that is not JSON-compatible.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def content_hash(obj: Any) -> str:
    """Return the versioned SHA-256 hash of ``obj``'s canonical JSON, as ``"fg1:<hex>"``."""
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return f"{HASH_SCHEME}:{digest}"


def short_hash(full: str, length: int = 12) -> str:
    """Return the first ``length`` hex digits of a hash produced by `content_hash`."""
    return full.split(":", 1)[-1][:length]
