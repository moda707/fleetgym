"""In-memory event log and its Parquet I/O."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from fleetgym.errors import FleetGymError
from fleetgym.eventlog.schema import EVENT_KINDS, LOG_SCHEMA, LOG_SCHEMA_VERSION

__all__ = ["EventLog"]

_COLUMNS = tuple(LOG_SCHEMA.names)
_FIELDS = frozenset(_COLUMNS) - {"seq", "kind"}


class EventLog:
    """An append-only list of events with metadata.

    Parameters
    ----------
    metadata
        String key-value pairs describing the episode (see `METADATA_KEYS`).
    rows
        Existing rows, for example from `read`.
    """

    def __init__(
        self, metadata: Mapping[str, str] | None = None, rows: list[dict[str, Any]] | None = None
    ) -> None:
        self.metadata: dict[str, str] = {"fleetgym_log_version": LOG_SCHEMA_VERSION}
        self.metadata.update(metadata or {})
        self.rows: list[dict[str, Any]] = rows if rows is not None else []

    def append(self, kind: str, **fields: Any) -> None:
        """Append an event; unknown kinds or columns raise immediately."""
        if kind not in EVENT_KINDS:
            raise FleetGymError(f"unknown event kind {kind!r}")
        unknown = fields.keys() - _FIELDS
        if unknown:
            raise FleetGymError(f"unknown event log column(s) {sorted(unknown)}")
        self.rows.append({"seq": len(self.rows), "kind": kind, **fields})

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.rows)

    def of_kind(self, *kinds: str) -> list[dict[str, Any]]:
        """Rows of the given kinds, in order."""
        return [row for row in self.rows if row["kind"] in kinds]

    def truncate(self, length: int) -> None:
        """Drop every row after the first ``length`` (used when restoring a snapshot)."""
        del self.rows[length:]

    def copy(self) -> EventLog:
        """Copy that shares no mutable state with this log (rows are not modified in place)."""
        return EventLog(dict(self.metadata), list(self.rows))

    def to_table(self) -> pa.Table:
        """The log as an Arrow table with `LOG_SCHEMA` and the metadata attached."""
        columns = {name: [row.get(name) for row in self.rows] for name in _COLUMNS}
        table = pa.Table.from_pydict(columns, schema=LOG_SCHEMA)
        return table.replace_schema_metadata({k: str(v) for k, v in self.metadata.items()})

    def write(self, path: str | Path) -> None:
        """Write the log to a Parquet file."""
        pq.write_table(self.to_table(), Path(path))

    @classmethod
    def read(cls, path: str | Path) -> EventLog:
        """Read a log written by `write`.

        Raises
        ------
        FleetGymError
            If the file was written with an incompatible log schema version.
        """
        table = pq.read_table(Path(path))
        raw = table.schema.metadata or {}
        metadata = {k.decode(): v.decode() for k, v in raw.items()}
        version = metadata.get("fleetgym_log_version")
        if version != LOG_SCHEMA_VERSION:
            raise FleetGymError(
                f"{path}: event log schema version {version!r}, expected {LOG_SCHEMA_VERSION!r}"
            )
        rows = [
            {k: v for k, v in row.items() if v is not None or k in ("seq", "kind")}
            for row in table.select(list(_COLUMNS)).to_pylist()
        ]
        return cls(metadata, rows)

    def digest(self) -> str:
        """SHA-256 of the rows (not the metadata), for determinism tests."""
        text = json.dumps(self.rows, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
