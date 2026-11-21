from pathlib import Path

import pytest

from fleetgym.cli.__main__ import main
from fleetgym.spec import load_spec
from fleetgym.spec.yaml_io import load_yaml


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_list(capsys):
    code, out, _ = run(capsys, "list")
    assert code == 0
    assert "CVRP-v1" in out and "deterministic" in out
    assert "EVRP-NL-v1" not in out
    _, out_all, _ = run(capsys, "list", "--all")
    assert "EVRP-NL-v1" in out_all and "planned" in out_all


def test_show_raw_and_with_defaults(capsys):
    code, out, _ = run(capsys, "show", "VRPTW-v1", "--set", "agents.mode=multi")
    assert code == 0
    raw = load_yaml(out)
    assert raw["id"] == "VRPTW-v1"
    assert raw["agents"] == {"mode": "multi"}
    assert "extends" not in raw
    code, out, _ = run(capsys, "show", "VRPTW-v1", "--defaults")
    assert code == 0
    first, rest = out.split("\n", 1)
    assert first == f"# spec hash: {load_spec('VRPTW-v1').content_hash()}"
    assert load_yaml(rest)["decisions"]["commitment"] == "next_stop"


def test_validate(capsys, tmp_path: Path):
    code, out, _ = run(capsys, "validate", "--all")
    assert code == 0
    assert out.count("OK ") == 7
    assert "reserved: ev, constraints.energy" in out
    bad = tmp_path / "bad.yaml"
    bad.write_text("extends: CVRP-v1\nobjective: {bogus: 1}\n")
    code, out, err = run(capsys, "validate", "CVRP-v1", str(bad))
    assert code == 1
    assert "OK   CVRP-v1" in out
    assert "FAIL" in err and "objective" in err
    code, _, err = run(capsys, "validate")
    assert code == 2 and "at least one" in err


def test_schema(capsys, tmp_path: Path):
    code, out, _ = run(capsys, "schema")
    assert code == 0 and '"title": "FleetGym config"' in out
    target = tmp_path / "s.json"
    assert run(capsys, "schema", "-o", str(target))[0] == 0
    assert target.read_text() == out


def test_errors_exit_with_code_1(capsys):
    code, _, err = run(capsys, "show", "NOPE-v0")
    assert code == 1
    assert err.startswith("error: unknown environment ID 'NOPE-v0'")
