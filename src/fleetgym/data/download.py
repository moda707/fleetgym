"""Explicit, checksummed downloads of benchmark collections.

Collections are described by manifests shipped in ``fleetgym/data/manifests/`` (built with the
``scripts/build_*_manifest.py`` maintainer tools). A file is either downloaded directly or extracted
from an archive the manifest lists under ``archives`` (the archive is verified, then the member).
Files are downloaded on request only, cached under
``$FLEETGYM_DATA_DIR`` (default ``~/.cache/fleetgym``), and verified against the manifest's SHA-256.
A checksum mismatch is an error: when a collection improves a best-known solution upstream, the
manifest must be rebuilt rather than silently accepting a different file.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from fleetgym.errors import InstanceError

__all__ = ["Collection", "cache_dir", "collections", "fetch", "load_collection"]

_USER_AGENT = "fleetgym-data"


@dataclass(frozen=True)
class Collection:
    """A benchmark collection described by a manifest.

    ``tolerance`` is the absolute difference allowed between a recomputed cost and the published
    one: 0 for integer costs, half a unit in the last published decimal for rounded values.
    """

    name: str
    base_url: str
    format: str
    items: dict[str, dict[str, Any]]
    retrieved: str
    archives: dict[str, dict[str, Any]] = field(default_factory=dict)
    tolerance: float = 0.0

    def names(self) -> tuple[str, ...]:
        """Instance names, sorted."""
        return tuple(sorted(self.items))

    def published_cost(self, name: str) -> float | None:
        """Cost of the best-known solution recorded in the manifest."""
        return self._item(name).get("cost")

    def info(self, name: str) -> dict[str, Any]:
        """The manifest entry of ``name`` (published values, reference, notes)."""
        return dict(self._item(name))

    def _item(self, name: str) -> dict[str, Any]:
        if name not in self.items:
            raise InstanceError(f"{self.name} has no instance {name!r}")
        return self.items[name]


def _manifest_files() -> dict[str, Path]:
    root = Path(str(resources.files("fleetgym.data"))) / "manifests"
    out = {}
    for path in sorted(root.glob("*.json")):
        name = json.loads(path.read_text(encoding="utf-8"))["collection"]
        out[name] = path
    return out


def collections() -> tuple[str, ...]:
    """Names of the collections with a shipped manifest, for example ``cvrplib/X``."""
    return tuple(_manifest_files())


def load_collection(name: str) -> Collection:
    """Read the manifest of collection ``name``."""
    files = _manifest_files()
    if name not in files:
        raise InstanceError(f"unknown collection {name!r}; known: {', '.join(files)}")
    data = json.loads(files[name].read_text(encoding="utf-8"))
    return Collection(
        name=data["collection"],
        base_url=data["base_url"],
        format=data["format"],
        items=data["items"],
        retrieved=data["retrieved"],
        archives=data.get("archives", {}),
        tolerance=float(data.get("tolerance", 0.0)),
    )


def cache_dir() -> Path:
    """Download cache: ``$FLEETGYM_DATA_DIR`` or ``~/.cache/fleetgym``."""
    return Path(os.environ.get("FLEETGYM_DATA_DIR", Path.home() / ".cache" / "fleetgym"))


def fetch(
    collection: str,
    name: str,
    *,
    kind: Literal["instance", "solution"] = "instance",
    cache: Path | None = None,
) -> Path:
    """Return the local path of a collection file, downloading and verifying it if needed.

    Raises
    ------
    InstanceError
        If the collection or instance is unknown, the download fails, or the checksum does not match
        the manifest.
    """
    coll = load_collection(collection)
    entry = coll._item(name)[kind]  # pyright: ignore[reportPrivateUsage]
    root = (cache or cache_dir()) / collection
    target = root / entry["file"]
    if target.exists() and _sha256(target.read_bytes()) == entry["sha256"]:
        return target
    label = f"{collection}/{entry['file']}"
    if "archive" in entry:
        if entry["archive"] not in coll.archives:
            raise InstanceError(f"{label}: archive {entry['archive']!r} is not in the manifest")
        archive = coll.archives[entry["archive"]]
        archive_path = root / entry["archive"]
        if not (archive_path.exists() and _sha256(archive_path.read_bytes()) == archive["sha256"]):
            _store(archive_path, _download(coll, archive["path"], archive["sha256"], label))
        try:
            with zipfile.ZipFile(archive_path) as zf:
                data = zf.read(entry["member"])
        except (KeyError, zipfile.BadZipFile) as exc:
            raise InstanceError(f"{label}: cannot extract {entry['member']!r}: {exc}") from exc
        if _sha256(data) != entry["sha256"]:
            raise InstanceError(f"{label}: checksum mismatch for archive member {entry['member']}")
    else:
        data = _download(coll, entry["path"], entry["sha256"], label)
    _store(target, data)
    return target


def _download(coll: Collection, path: str, sha256: str, label: str) -> bytes:
    url = coll.base_url + path
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    except OSError as exc:
        raise InstanceError(f"downloading {url} failed: {exc}") from exc
    if _sha256(data) != sha256:
        raise InstanceError(
            f"{label}: checksum mismatch; the file changed upstream since "
            f"{coll.retrieved}. Rebuild the manifest after checking the change."
        )
    return data


def _store(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
