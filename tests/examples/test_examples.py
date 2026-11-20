"""Every example runs (the documentation includes them verbatim)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
NEEDS = {"06_train_agent.py": "sb3_contrib", "08_render.py": "matplotlib"}
SCRIPTS = [*sorted(EXAMPLES.glob("0*.py")), EXAMPLES / "05_custom_variant" / "run.py"]


@pytest.mark.parametrize(
    "script", SCRIPTS, ids=lambda p: p.parent.name if p.name == "run.py" else p.stem
)
def test_example_runs(script: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if script.name in NEEDS:
        pytest.importorskip(NEEDS[script.name])
    monkeypatch.chdir(tmp_path)  # outputs land in a temporary runs/
    monkeypatch.setattr(sys, "argv", [str(script)])
    runpy.run_path(str(script), run_name="__main__")


def test_custom_plugins_are_cleaned_up():
    """The custom variant registers global plugins; remove them so other tests see the defaults."""
    from fleetgym.constraints.base import CONSTRAINT_CLASSES
    from fleetgym.sim.recourse import RECOURSE_CLASSES
    from fleetgym.spec.registries import CONSTRAINTS, DISTRIBUTIONS, RECOURSE_RULES
    from fleetgym.uncertainty.distributions import DISTRIBUTION_CLASSES

    for registry, name in [
        (DISTRIBUTIONS, "triangular_demand"),
        (DISTRIBUTION_CLASSES, "triangular_demand"),
        (CONSTRAINTS, "max_stops"),
        (CONSTRAINT_CLASSES, "max_stops"),
        (RECOURSE_RULES, "near_depot_restock"),
        (RECOURSE_CLASSES, "near_depot_restock"),
    ]:
        if name in registry:
            registry.unregister(name)
    sys.modules.pop("plugins", None)
