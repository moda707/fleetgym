import json
from pathlib import Path

from fleetgym.testing.golden import registered_spec_hashes

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "spec_hashes.json"


def test_registered_spec_hashes_match_snapshot(update_golden: bool):
    current = registered_spec_hashes()
    if update_golden or not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    stored = json.loads(GOLDEN.read_text())
    changed = sorted(k for k in current.keys() & stored.keys() if current[k] != stored[k])
    assert not changed, (
        f"spec hashes changed for {changed}. If dynamics or defaults changed on purpose, bump the "
        "env version (-vN); otherwise rerun with --update-golden."
    )
    assert current.keys() == stored.keys(), "registered IDs changed; rerun with --update-golden"
