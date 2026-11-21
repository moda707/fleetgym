"""Instance sets: a folder of FleetGym instance files with a manifest (ADR 0023).

A set holds many instances of one schema, such as a paper's test set or a user's training data
(FleetGym ships no instance generators: producing instances is the user's task). The manifest
``manifest.json`` records the schema, the files with their SHA-256 checksums, and where the
instances come from::

    {"format": "fleetgym-instance-set/1", "name": "...", "schema": "fleetgym/core/1",
     "reference": "...", "produced_by": "...", "files": {"0001.json": "<sha256>", ...}}

`read_instance_set` verifies every checksum, so results computed on a set name exactly the data
they used (`InstanceSet.content_hash`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetgym._hashing import content_hash
from fleetgym.data.instance import Instance
from fleetgym.data.readers.fleetgym_json import read_fleetgym_json, write_instance
from fleetgym.errors import InstanceError

__all__ = ["MANIFEST", "SET_FORMAT", "InstanceSet", "read_instance_set", "write_instance_set"]

SET_FORMAT = "fleetgym-instance-set/1"
MANIFEST = "manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class InstanceSet:
    """A verified instance set.

    Attributes
    ----------
    root
        The folder.
    name
        The set's name.
    schema
        The schema every instance follows.
    files
        Instance file names (relative to ``root``) and their checksums, in manifest order.
    info
        The rest of the manifest (``reference``, ``produced_by``, ...).
    """

    root: Path
    name: str
    schema: str
    files: tuple[tuple[str, str], ...]
    info: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.files)

    @property
    def paths(self) -> tuple[Path, ...]:
        """Paths of the instance files."""
        return tuple(self.root / name for name, _ in self.files)

    def load(self, index: int) -> Instance:
        """Read instance ``index``."""
        return read_fleetgym_json(self.paths[index])

    def __iter__(self) -> Iterator[Instance]:
        return (self.load(i) for i in range(len(self)))

    @property
    def max_nodes(self) -> int:
        """The largest instance's node count (recorded in the manifest, else computed)."""
        recorded = self.info.get("max_nodes")
        if isinstance(recorded, int):
            return recorded
        return max(instance.n_nodes for instance in self)

    @property
    def max_vehicles(self) -> int:
        """The largest fleet (sum of vehicle type counts), from the manifest or computed."""
        recorded = self.info.get("max_vehicles")
        if isinstance(recorded, int):
            return recorded
        return max(sum(int(vt.get("count", 0)) for vt in i.vehicle_types) for i in self)

    def content_hash(self) -> str:
        """Hash of the schema and every file checksum."""
        return content_hash({"schema": self.schema, "files": [list(f) for f in self.files]})


def write_instance_set(
    instances: Iterable[Instance],
    folder: str | Path,
    *,
    name: str,
    reference: str = "",
    produced_by: str = "",
    **info: Any,
) -> InstanceSet:
    """Write instances as ``0001.json``, ``0002.json``, ... and a manifest into ``folder``.

    Raises
    ------
    InstanceError
        If the instances do not all follow one schema, or the folder already holds a set.
    """
    root = Path(folder)
    if (root / MANIFEST).exists():
        raise InstanceError(f"{root}: already holds an instance set")
    root.mkdir(parents=True, exist_ok=True)
    files: list[tuple[str, str]] = []
    schema: str | None = None
    max_nodes = max_vehicles = 0
    for index, instance in enumerate(instances, start=1):
        max_nodes = max(max_nodes, instance.n_nodes)
        fleet = sum(int(vt.get("count", 0)) for vt in instance.vehicle_types)
        max_vehicles = max(max_vehicles, fleet)
        if schema is None:
            schema = instance.schema
        elif instance.schema != schema:
            raise InstanceError(
                f"instance {index} follows {instance.schema!r}, the set follows {schema!r}"
            )
        path = write_instance(instance, root / f"{index:04d}.json")
        files.append((path.name, _sha256(path)))
    if schema is None:
        raise InstanceError("an instance set needs at least one instance")
    manifest = {
        "format": SET_FORMAT,
        "name": name,
        "schema": schema,
        "reference": reference,
        "produced_by": produced_by,
        "max_nodes": max_nodes,
        "max_vehicles": max_vehicles,
        **info,
        "files": dict(files),
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return read_instance_set(root)


def read_instance_set(folder: str | Path) -> InstanceSet:
    """Read a set's manifest and verify every file's checksum.

    Raises
    ------
    InstanceError
        If the manifest is missing or malformed, a file is missing, or a checksum does not match.
    """
    root = Path(folder)
    path = root / MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstanceError(f"{path}: cannot read the instance set manifest: {exc}") from exc
    if manifest.get("format") != SET_FORMAT:
        raise InstanceError(f'{path}: not an instance set (expected "format": "{SET_FORMAT}")')
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise InstanceError(f"{path}: `files` must map file names to SHA-256 checksums")
    for name, checksum in files.items():
        file = root / name
        if not file.is_file():
            raise InstanceError(f"{path}: listed file {name!r} is missing")
        if _sha256(file) != checksum:
            raise InstanceError(f"{path}: {name!r} does not match its checksum (it was changed)")
    info = {k: v for k, v in manifest.items() if k not in ("format", "name", "schema", "files")}
    return InstanceSet(
        root=root,
        name=str(manifest.get("name", root.name)),
        schema=str(manifest.get("schema", "fleetgym/core/1")),
        files=tuple(files.items()),
        info=info,
    )
