"""Build the checksummed manifest of a CVRPLIB instance set (maintainer tool).

Usage::

    uv run python -I scripts/build_cvrplib_manifest.py X src/fleetgym/data/manifests/cvrplib_X.json

Reads the CVRPLIB index page, downloads every instance and solution file of the set once, and writes
their download paths, SHA-256 checksums and published costs. `fleetgym.data.download.fetch` verifies
files against this manifest. Checksums record the files as published on the build date; when
CVRPLIB improves a best-known solution, its checksum changes and the manifest must be rebuilt.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://galgos.inf.puc-rio.br/cvrplib/index.php/en/"
DOWNLOAD = BASE + "download/"
USER_AGENT = "fleetgym-manifest-builder"


def get(url: str) -> bytes:
    """Download ``url``."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main(set_name: str, output: Path) -> None:
    """Write the manifest of CVRPLIB set ``set_name`` to ``output``."""
    html = get(BASE).decode("utf-8", errors="replace")
    rows = re.findall(
        r'download/instance/(\d+)" title="Instance File">\s*([A-Za-z0-9-]+)\s*</a>'
        r'.*?download/bks/(\d+)"',
        html,
        flags=re.S,
    )
    items: dict[str, dict[str, object]] = {}
    for instance_id, name, bks_id in rows:
        if not name.startswith(f"{set_name}-"):
            continue
        instance = get(f"{DOWNLOAD}instance/{instance_id}")
        solution = get(f"{DOWNLOAD}bks/{bks_id}")
        cost = re.search(rb"^Cost\s+([0-9.]+)", solution, flags=re.M)
        items[name] = {
            "instance": {
                "path": f"instance/{instance_id}",
                "file": f"{name}.vrp",
                "sha256": hashlib.sha256(instance).hexdigest(),
            },
            "solution": {
                "path": f"bks/{bks_id}",
                "file": f"{name}.sol",
                "sha256": hashlib.sha256(solution).hexdigest(),
            },
            "cost": float(cost.group(1)) if cost else None,
        }
        print(name, flush=True)
        time.sleep(0.2)
    if not items:
        raise SystemExit(f"no instances of set {set_name!r} found on {BASE}")
    manifest = {
        "collection": f"cvrplib/{set_name}",
        "source": "CVRPLIB, http://vrp.galgos.inf.puc-rio.br",
        "base_url": DOWNLOAD,
        "retrieved": dt.date.today().isoformat(),
        "format": "vrplib",
        "solution_node_indexing": "customer k in a solution file is node k (depot = node 0)",
        "items": dict(sorted(items.items())),
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(items)} instances -> {output}")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
