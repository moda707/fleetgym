"""Per-source random streams derived from one episode seed (architecture rule 5, ADR 0009).

Each source of randomness gets its own generator, keyed by a hash of the source's *name*:
``SeedSequence(entropy=episode_seed, spawn_key=(source_id,))``. Adding, removing or reordering
sources never changes the draws of the others; `numpy.random.SeedSequence.spawn` is not used
because its children depend on spawn order.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["StreamFactory", "source_id"]

_UINT32 = 2**32


def source_id(name: str) -> int:
    """Stable 32-bit identifier of a source name (first 4 bytes of its SHA-256, little-endian)."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


class StreamFactory:
    """Independent random generators for named sources of one episode.

    Parameters
    ----------
    episode_seed
        Non-negative integer seed of the episode.

    Examples
    --------
    >>> streams = StreamFactory(7)
    >>> a = streams.generator("demand").random()
    >>> b = StreamFactory(7).generator("demand").random()
    >>> a == b
    True
    """

    def __init__(self, episode_seed: int) -> None:
        if episode_seed < 0:
            raise ValueError(f"episode seeds must be non-negative, got {episode_seed}")
        self.episode_seed = int(episode_seed)

    def seed_sequence(self, source: str, *key: int) -> np.random.SeedSequence:
        """Seed sequence of ``source``; extra integers in ``key`` give sub-streams."""
        return np.random.SeedSequence(
            entropy=self.episode_seed, spawn_key=(source_id(source), *(k % _UINT32 for k in key))
        )

    def generator(self, source: str, *key: int) -> np.random.Generator:
        """A fresh generator for ``source`` (the same arguments always give the same stream)."""
        return np.random.Generator(np.random.PCG64(self.seed_sequence(source, *key)))
