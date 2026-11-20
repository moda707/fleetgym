"""Deterministic train, validation and test scenario sets (design §23, common random numbers).

The seeds of a split are derived from the environment ID and the split name, so every policy faces
the same scenarios, and splits do not overlap (checked). Test seeds must never be used for training
or tuning.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np

__all__ = ["Split", "scenario_seeds"]

Split = Literal["train", "validation", "test"]
_SEED_LIMIT = 2**63


def _seeds(env_id: str, split: str, n: int) -> list[int]:
    entropy = int.from_bytes(hashlib.sha256(f"{env_id}|{split}".encode()).digest()[:16], "little")
    state = np.random.SeedSequence(entropy).generate_state(n, np.uint64)
    return [int(s) % _SEED_LIMIT for s in state]


def scenario_seeds(env_id: str, split: Split, n: int) -> list[int]:
    """The first ``n`` episode seeds of ``split`` for ``env_id``.

    Raises
    ------
    ValueError
        If ``n`` is not positive or the split is unknown.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split {split!r}")
    seeds = _seeds(env_id, split, n)
    others = {
        s
        for other in ("train", "validation", "test")
        if other != split
        for s in _seeds(env_id, other, n)
    }
    if others & set(seeds):  # pragma: no cover - probability ~ n^2 / 2^63
        raise RuntimeError("scenario splits overlap")
    return seeds
