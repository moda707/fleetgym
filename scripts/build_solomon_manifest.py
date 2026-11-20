"""Build the checksummed manifest of the Solomon 100-customer VRPTW set (maintainer tool).

Usage::

    uv run python -I scripts/build_solomon_manifest.py src/fleetgym/data/manifests/solomon_100.json

Source: SINTEF TOP, https://www.sintef.no/projectweb/top/vrptw/100-customers/ . Instances come from
the zip archive linked there; best-known solutions are the per-instance route files linked from the
table. The table's vehicles, distance (rounded to two decimals), reference and comment are copied
into the manifest. SINTEF's convention, as stated on the page: hierarchical objective (vehicles,
then distance), Euclidean distance equals travel time, double precision, totals rounded to two
decimals. A trailing ``*`` on a distance marks a corrected value explained in the page's footnote.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

SITE = "https://www.sintef.no/"
PAGE = SITE + "projectweb/top/vrptw/100-customers/"
ARCHIVE = "globalassets/project/top/vrptw/solomon/solomon-100.zip"
USER_AGENT = "fleetgym-manifest-builder"
ROW = re.compile(
    r'<tr>\s*<td[^>]*><a href="/(contentassets/[^"]+\.txt)"[^>]*>([^<]+)</a></td>'
    r"\s*<td[^>]*>([^<]*)</td>\s*<td[^>]*>([^<]*)</td>\s*<td[^>]*>([^<]*)</td>"
    r"\s*<td[^>]*>(.*?)</td>",
    flags=re.S,
)


def get(url: str) -> bytes:
    """Download ``url``."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def text(fragment: str) -> str:
    """Plain text of an HTML table cell."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def main(output: Path) -> None:
    """Write the Solomon 100-customer manifest to ``output``."""
    page = get(PAGE).decode("utf-8", errors="replace")
    archive = get(SITE + ARCHIVE)
    members = zipfile.ZipFile(io.BytesIO(archive))
    items: dict[str, dict[str, object]] = {}
    for path, label, vehicles, distance, reference, comment in ROW.findall(page):
        # The instance is named after the solution file (the table shows e.g. "r205b" for r205).
        name = Path(path).stem.split("_")[0].lower()
        member = f"In/{name}.txt"
        solution = get(SITE + path)
        note = text(comment)
        items[name] = {
            "instance": {
                "file": f"instances/{name}.txt",
                "archive": "solomon-100.zip",
                "member": member,
                "sha256": hashlib.sha256(members.read(member)).hexdigest(),
            },
            "solution": {
                "path": path,
                "file": f"solutions/{name}.txt",
                "sha256": hashlib.sha256(solution).hexdigest(),
            },
            "vehicles": int(text(vehicles)),
            "cost": float(text(distance).rstrip("*")),
            "reference": text(reference),
            "table_label": text(label),
            "note": note if note and not note.startswith("Detailed solution") else None,
        }
        print(name, flush=True)
        time.sleep(0.2)
    if len(items) != 56:
        raise SystemExit(f"expected 56 instances on {PAGE}, found {len(items)}")
    manifest = {
        "collection": "solomon/100",
        "source": "SINTEF TOP, " + PAGE,
        "base_url": SITE,
        "retrieved": dt.date.today().isoformat(),
        "format": "solomon",
        "conventions": (
            "hierarchical objective: 1) number of vehicles, 2) total distance; Euclidean distance "
            "equals travel time; double precision; published totals rounded to two decimals"
        ),
        "tolerance": 0.005,
        "solution_node_indexing": "customer k in a solution file is node k (depot = node 0)",
        "archives": {
            "solomon-100.zip": {"path": ARCHIVE, "sha256": hashlib.sha256(archive).hexdigest()}
        },
        "items": dict(sorted(items.items())),
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(items)} instances -> {output}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
