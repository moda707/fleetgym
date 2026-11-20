from pathlib import Path

import pytest

import fleetgym.cli.replicate as replicate
from fleetgym.cli.__main__ import main
from fleetgym.data.download import Collection

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "instances"


def fake_collection(cost: float) -> Collection:
    return Collection(
        name="cvrplib/X",
        base_url="",
        format="vrplib",
        items={"tiny-n5": {"cost": cost}},
        retrieved="",
    )


@pytest.mark.parametrize(("published", "code", "word"), [(17.0, 0, "OK"), (18.0, 1, "FAIL")])
def test_replicate_with_fixture(monkeypatch, capsys, published, code, word, tmp_path: Path):
    sol = tmp_path / "tiny.sol"
    sol.write_text((FIXTURES / "tiny.sol").read_text().replace("Cost 17", f"Cost {published:g}"))
    monkeypatch.setattr(replicate, "load_collection", lambda name: fake_collection(published))
    monkeypatch.setattr(
        replicate,
        "fetch",
        lambda coll, name, kind="instance": FIXTURES / "tiny.vrp" if kind == "instance" else sol,
    )
    assert main(["replicate", "CVRP-v0", "--limit", "1"]) == code
    out, err = capsys.readouterr()
    assert "tiny-n5" in out and word in out
    assert f"{1 - code}/1 instances" in err


def test_replicate_reports_infeasible_solutions(monkeypatch, capsys, tmp_path: Path):
    sol = tmp_path / "bad.sol"
    sol.write_text("Route #1: 1 2 3 4\nCost 17\n")  # demand 14 > capacity 10
    monkeypatch.setattr(replicate, "load_collection", lambda name: fake_collection(17.0))
    monkeypatch.setattr(
        replicate,
        "fetch",
        lambda coll, name, kind="instance": FIXTURES / "tiny.vrp" if kind == "instance" else sol,
    )
    assert main(["replicate", "CVRP-v0", "--instances", "tiny-n5"]) == 1
    assert "capacity" in capsys.readouterr().out


def test_collection_for():
    assert replicate.collection_for("CVRP-v0") == "cvrplib/X"
    assert main(["replicate", "VRPTW-v0"]) == 1
