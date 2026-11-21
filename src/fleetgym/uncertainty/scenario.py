"""`Scenario`: every exogenous realization of one episode (design §6, architecture rule 6).

A scenario holds, for each uncertainty source of the spec (for example ``demand``), one realized
value per node. With dynamic requests it also holds the requests of the day (``request.*`` sources,
one value per node slot; `episode_instance` turns them into the episode's nodes) and the release
times of requests beyond ``max_requests`` (``request.overflow``, any length). It is sampled once at
reset (`fleetgym.uncertainty.sampling`), replayed exactly, and stored as Parquet with the spec and
instance hashes it belongs to. The simulator reads true values from it; the agent only sees what has
been revealed.

This module depends only on layers L0 to L2, so the independent validator may import it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from fleetgym.data.instance import Instance
from fleetgym.errors import ScenarioMismatchError

__all__ = ["REQUEST_SOURCES", "Scenario", "episode_instance"]

_META_KEY = b"fleetgym.scenario"

#: Per-node-slot sources of dynamic requests (static customers included, absent slots at the end).
REQUEST_SOURCES = (
    "request.release",
    "request.x",
    "request.y",
    "request.demand",
    "request.tw_open",
    "request.tw_close",
    "request.service_time",
    "request.cancel",
)
_VARIABLE_LENGTH = ("request.overflow",)


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
            if name not in _VARIABLE_LENGTH:
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
        """Content hash of the scenario (hashes and values; the seed is not part of it).

        Values are hashed as little-endian float64 bytes, so the id is exact and covers ``inf``.
        """
        digest = hashlib.sha256()
        digest.update(f"{self.spec_hash}|{self.instance_hash}".encode())
        for name, arr in self.values.items():
            digest.update(f"|{name}:{arr.size}|".encode())
            digest.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
        return "fg1:" + digest.hexdigest()

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
            if name not in _VARIABLE_LENGTH and arr.size != n_nodes:
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
        lengths = {name: int(arr.size) for name, arr in self.values.items()}
        rows = max(lengths.values(), default=0)
        columns = {
            name: pa.array(np.pad(arr, (0, rows - arr.size), constant_values=np.nan), pa.float64())
            for name, arr in self.values.items()
        }
        table = pa.table(columns) if columns else pa.table({"_empty": pa.array([], pa.int8())})
        meta = {
            "spec_hash": self.spec_hash,
            "instance_hash": self.instance_hash,
            "seed": self.seed,
            "lengths": lengths,
        }
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
        lengths = meta.get("lengths", {})
        values = {
            name: table.column(name).to_numpy()[: lengths.get(name, table.num_rows)]
            for name in table.column_names
            if name != "_empty"
        }
        return cls(meta["spec_hash"], meta["instance_hash"], values, seed=meta["seed"])


def episode_instance(base: Instance, scenario: Scenario) -> Instance:
    """The instance of one episode: ``base`` plus the scenario's dynamic request slots.

    Without ``request.*`` sources the base instance is returned unchanged. Otherwise node ``i`` of
    the result has the scenario's location, demand, time window and service time of slot ``i``;
    depots keep the base instance's values. ``extra`` records ``_release`` and ``_cancel`` times
    per node, ``_n_static`` (nodes known at the start), ``_n_slots`` and ``_base_instance_hash``.
    """
    if "request.release" not in scenario.values:
        return base
    per_node = sorted(k for k, v in base.extra.items() if not k.startswith("_") and np.ndim(v) > 0)
    if per_node:
        from fleetgym.errors import NotInScopeError

        raise NotInScopeError(
            f"instance {base.name!r} has node or arc fields {per_node}; requests arriving during "
            "the day have no values for them (not supported in this version)"
        )
    release = scenario["request.release"]
    n = release.size
    depots = base.n_depots
    coords = np.column_stack([scenario["request.x"], scenario["request.y"]])
    demands = scenario["request.demand"].copy()
    service = scenario["request.service_time"].copy()
    tw_open, tw_close = scenario["request.tw_open"], scenario["request.tw_close"]
    windows = None
    if base.time_windows is not None:
        windows = np.column_stack([tw_open, tw_close])
        windows[:depots] = base.time_windows[:depots]
    if base.coords is not None:
        coords[:depots] = base.coords[:depots]
    demands[:depots] = 0.0
    if base.service_times is not None:
        service[:depots] = base.service_times[:depots]
    static = int(np.sum(release <= release[0]))
    return Instance(
        name=f"{base.name}+requests",
        source=base.source,
        demands=demands,
        coords=coords,
        capacity=base.capacity,
        n_vehicles=base.n_vehicles,
        time_windows=windows,
        service_times=service,
        n_depots=depots,
        schema=base.schema,
        distributions=base.distributions,
        vehicle_types=base.vehicle_types,
        extra={
            **{k: v for k, v in base.extra.items() if np.ndim(v) == 0},
            "_release": release,
            "_cancel": scenario["request.cancel"],
            "_n_static": static,
            "_n_slots": n,
            "_base_instance_hash": base.content_hash(),
        },
    )
