import hashlib
import io
import json
from pathlib import Path

import pytest

import fleetgym.data.download as download
from fleetgym.errors import InstanceError


def fake_manifest(tmp_path: Path, payload: bytes) -> Path:
    root = tmp_path / "pkg" / "manifests"
    root.mkdir(parents=True)
    manifest = {
        "collection": "test/set",
        "source": "test",
        "base_url": "https://example.invalid/",
        "retrieved": "2026-11-20",
        "format": "vrplib",
        "items": {
            "A": {
                "instance": {
                    "path": "inst/1",
                    "file": "A.vrp",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "solution": {"path": "sol/1", "file": "A.sol", "sha256": "0" * 64},
                "cost": 42.0,
            }
        },
    }
    (root / "test.json").write_text(json.dumps(manifest))
    return root.parent


@pytest.fixture
def patched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = b"NAME : A\n"
    pkg = fake_manifest(tmp_path, payload)
    monkeypatch.setattr(download.resources, "files", lambda _: pkg)
    calls: list[str] = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
            return None

    def urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("sol/1"):
            return Response(b"Route #1: 1\nCost 42\n")
        return Response(payload)

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("FLEETGYM_DATA_DIR", str(tmp_path / "cache"))
    return calls


def test_fetch_downloads_verifies_and_caches(patched, tmp_path: Path):
    assert download.collections() == ("test/set",)
    coll = download.load_collection("test/set")
    assert coll.names() == ("A",)
    assert coll.published_cost("A") == 42.0
    path = download.fetch("test/set", "A")
    assert path == tmp_path / "cache" / "test/set" / "A.vrp"
    assert path.read_bytes() == b"NAME : A\n"
    download.fetch("test/set", "A")
    assert patched == ["https://example.invalid/inst/1"]  # second call served from the cache


def test_checksum_mismatch_and_unknown_names(patched):
    with pytest.raises(InstanceError, match="checksum mismatch"):
        download.fetch("test/set", "A", kind="solution")
    with pytest.raises(InstanceError, match="no instance 'B'"):
        download.fetch("test/set", "B")
    with pytest.raises(InstanceError, match="unknown collection"):
        download.load_collection("nope")


def test_network_errors_are_reported(patched, monkeypatch: pytest.MonkeyPatch):
    def broken(request, timeout):
        raise OSError("no route to host")

    monkeypatch.setattr(download.urllib.request, "urlopen", broken)
    with pytest.raises(InstanceError, match="no route to host"):
        download.fetch("test/set", "A")


def test_default_cache_dir(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FLEETGYM_DATA_DIR", raising=False)
    assert download.cache_dir() == Path.home() / ".cache" / "fleetgym"


def test_shipped_cvrplib_manifest():
    coll = download.load_collection("cvrplib/X")
    assert len(coll.names()) == 100
    assert coll.published_cost("X-n101-k25") == 27591.0
