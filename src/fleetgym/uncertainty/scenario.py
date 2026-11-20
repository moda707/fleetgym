"""`Scenario`: every exogenous realization of one episode (design §6, architecture rule 6).

A scenario holds, for each uncertainty source of the spec (for example ``demand``), one realized
value per node. It is sampled once at reset (`fleetgym.uncertainty.sampling`), replayed exactly,
and stored as Parquet with the spec and instance hashes it belongs to. The simulator reads true
values from it; the agent only sees what the revelation rules have revealed.

This module depends only on layers L0 to L2, so the independent validator may import it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from fleetgym._hashing import content_hash
from fleetgym.errors import ScenarioMismatchError

__all__ = ["Scenario"]

_META_KEY = b"fleetgym.scenario"


@dataclass(frozen=True, eq=False)
class Scenario:
    """Realized values of an episode's uncertainty sources.

    Attributes
    ----------
    spec_hash
        Content hash of the spec the scenario was sampled for.
    instance_hash
        Content hash of the instance.
    values
        Source name to a read-only array with one value per node.
    seed
        Episode seed it was sampled from, if any.
    """

    spec_hash: str
    instance_hash: str
    values: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    seed: int | None = None

    def __post_init__(self) -> None:
        frozen: dict[str, NDArray[np.float64]] = {}
        sizes = set()
        for name, array in self.values.items():
            arr = np.array(array, dtype=float)
            if arr.ndim != 1:
                raise ValueError(f"scenario values of {name!r} must be one-dimensional")
            arr.setflags(write=False)
            frozen[name] = arr
            sizes.add(arr.size)
        if len(sizes) > 1:
            raise ValueError(f"scenario sources have different lengths: {sorted(sizes)}")
        object.__setattr__(self, "values", MappingProxyType(dict(sorted(frozen.items()))))

    @property
    def sources(self) -> tuple[str, ...]:
        """Source names, sorted."""
        return tuple(self.values)

    @property
    def id(self) -> str:
        """Content hash of the scenario (hashes and values; the seed is not part of it)."""
        return content_hash(
            {
                "spec_hash": self.spec_hash,
                "instance_hash": self.instance_hash,
                "values": {k: v.tolist() for k, v in self.values.items()},
            }
        )

    def __getitem__(self, source: str) -> NDArray[np.float64]:
        return self.values[source]

    def check(
        self, *, spec_hash: str, instance_hash: str, sources: Iterable[str], n_nodes: int
    ) -> None:
        """Raise `ScenarioMismatchError` unless the scenario belongs to this spec and instance."""
        if self.spec_hash != spec_hash:
            raise ScenarioMismatchError(
                f"scenario was sampled for spec {self.spec_hash}, not {spec_hash}"
            )
        if self.instance_hash != instance_hash:
            raise ScenarioMismatchError(
                f"scenario was sampled for instance {self.instance_hash}, not {instance_hash}"
            )
        expected = tuple(sorted(sources))
        if self.sources != expected:
            raise ScenarioMismatchError(
                f"scenario has sources {list(self.sources)}, the spec has {list(expected)}"
            )
        for name, arr in self.values.items():
            if arr.size != n_nodes:
                raise ScenarioMismatchError(
                    f"scenario source {name!r} has {arr.size} values for {n_nodes} nodes"
                )

    def replace(self, **values: NDArray[np.float64]) -> Scenario:
        """A copy with some sources replaced (the seed is dropped)."""
        return Scenario(self.spec_hash, self.instance_hash, {**self.values, **values})

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Scenario):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    # ------------------------------------------------------------------ Parquet

    def to_table(self) -> pa.Table:
        """Arrow table: one row per node, one column per source, metadata in the schema."""
        columns = {name: pa.array(arr, type=pa.float64()) for name, arr in self.values.items()}
        table = pa.table(columns) if columns else pa.table({"_empty": pa.array([], pa.int8())})
        meta = {"spec_hash": self.spec_hash, "instance_hash": self.instance_hash, "seed": self.seed}
        return table.replace_schema_metadata({_META_KEY: json.dumps(meta).encode()})

    def write(self, path: str | Path) -> Path:
        """Write the scenario to a Parquet file and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(self.to_table(), path)
        return path

    @classmethod
    def read(cls, path: str | Path) -> Scenario:
        """Read a scenario written by `write`."""
        table = pq.read_table(path)
        raw = (table.schema.metadata or {}).get(_META_KEY)
        if raw is None:
            raise ScenarioMismatchError(f"{path} is not a FleetGym scenario file")
        meta = json.loads(raw)
        values = {
            name: table.column(name).to_numpy() for name in table.column_names if name != "_empty"
        }
        return cls(meta["spec_hash"], meta["instance_hash"], values, seed=meta["seed"])
