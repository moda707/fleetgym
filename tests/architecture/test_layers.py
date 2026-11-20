from pathlib import Path

import pytest

from tests.architecture.layers import LAYERS, find_violations

SRC = Path(__file__).resolve().parents[2] / "src"


def test_fleetgym_respects_layers():
    violations = find_violations(SRC)
    assert not violations, "\n".join(map(str, violations))


def test_every_top_level_package_has_a_layer():
    package = SRC / "fleetgym"
    tops = {p.stem for p in package.glob("*.py") if p.stem != "__init__"}
    tops |= {p.name for p in package.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    assert tops <= set(LAYERS), f"unassigned: {sorted(tops - set(LAYERS))}"


# ------------------------------------------------------------------ checker self-tests


def _make_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    src = tmp_path / "src"
    for rel, content in files.items():
        path = src / "fleetgym" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for directory in [p for p in (src / "fleetgym").rglob("*") if p.is_dir()]:
        (directory / "__init__.py").touch()
    (src / "fleetgym" / "__init__.py").touch()
    return src


@pytest.mark.parametrize(
    ("files", "expected_reason"),
    [
        ({"spec/a.py": "from fleetgym.sim import engine\n"}, "upward import"),
        ({"spec/a.py": "from ..sim import engine\n"}, "upward import"),
        ({"data/a.py": "import fleetgym.spec.loader\n"}, "upward import"),
        ({"network/a.py": "from fleetgym.uncertainty import scenario\n"}, "same-layer"),
        ({"constraints/a.py": "from fleetgym.sim import engine\n"}, "same-layer"),
        ({"interfaces/a.py": "from fleetgym.envs import env\n"}, "same-layer"),
        ({"testing/validator.py": "from fleetgym.network import euclidean\n"}, "independent"),
        ({"testing/validator.py": "from fleetgym.sim.state import SimState\n"}, "independent"),
        ({"sim/a.py": "import torch\n"}, "heavy optional dependency"),
        ({"envs/a.py": "def f():\n    import matplotlib.pyplot\n"}, "heavy optional dependency"),
        ({"spec/a.py": "import fleetgym\n"}, "facade"),
    ],
)
def test_checker_flags_violations(tmp_path: Path, files: dict[str, str], expected_reason: str):
    violations = find_violations(_make_tree(tmp_path, files))
    assert len(violations) == 1, violations
    assert expected_reason in violations[0].reason


@pytest.mark.parametrize(
    "files",
    [
        {"sim/a.py": "from fleetgym.spec import loader\nfrom fleetgym.errors import ConfigError\n"},
        {"sim/engine.py": "from fleetgym.constraints import base\n"},
        {"constraints/a.py": "from fleetgym.sim.state import SimState\n"},
        {"envs/a.py": "from fleetgym.interfaces import base\n"},
        {
            "testing/validator.py": (
                "from fleetgym.spec import loader\n"
                "from fleetgym.uncertainty.scenario import Scenario\n"
            )
        },
        {"nn/a.py": "import torch\n"},
        {"spec/a.py": "from . import b\nfrom .b import x\n", "spec/b.py": "x = 1\n"},
        {"plugins.py": "from fleetgym.errors import FleetGymError\n"},
    ],
)
def test_checker_accepts_allowed_imports(tmp_path: Path, files: dict[str, str]):
    assert find_violations(_make_tree(tmp_path, files)) == []
