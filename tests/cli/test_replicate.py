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


def test_collection_for(tmp_path: Path, capsys):
    assert replicate.collection_for("CVRP-v0") == "cvrplib/X"
    assert replicate.collection_for("VRPTW-v0") == "solomon/100"
    config = tmp_path / "Mine-v0.yaml"
    config.write_text("id: Mine-v0\nextends: CVRP-v0\n")  # meta is not inherited
    assert main(["replicate", str(config)]) == 1
    assert "no replication data" in capsys.readouterr().err


def test_published_cost_falls_back_to_the_manifest(monkeypatch, capsys, tmp_path: Path):
    sol = tmp_path / "nocost.sol"
    sol.write_text("Route #1: 1 3\nRoute #2: 2 4\n")
    coll = Collection(
        name="solomon/100",
        base_url="",
        format="vrplib",
        items={"tiny-n5": {"cost": 17.004}},
        retrieved="",
        tolerance=0.005,
    )
    monkeypatch.setattr(replicate, "load_collection", lambda name: coll)
    monkeypatch.setattr(
        replicate,
        "fetch",
        lambda coll, name, kind="instance": FIXTURES / "tiny.vrp" if kind == "instance" else sol,
    )
    assert main(["replicate", "CVRP-v0"]) == 0
    assert "17.004" in capsys.readouterr().out
    assert main(["replicate", "CVRP-v0", "--tolerance", "0"]) == 1
