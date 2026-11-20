"""Every example runs, as a script and as its notebook (the documentation includes the scripts)."""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_example_notebooks", ROOT / "scripts" / "build_example_notebooks.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _builder()
SCRIPTS: list[Path] = BUILDER.example_scripts()
NEEDS = {"06_train_agent": "sb3_contrib", "08_render": "matplotlib"}


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run in a scratch directory and remove the custom variant's global plugins afterwards."""
    monkeypatch.chdir(tmp_path)  # outputs land in a temporary runs/
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield
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


def _needs(script: Path) -> None:
    if script.parent.name in NEEDS:
        pytest.importorskip(NEEDS[script.parent.name])


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.parent.name)
def test_script_runs(script: Path, monkeypatch: pytest.MonkeyPatch):
    _needs(script)
    monkeypatch.setattr(sys, "argv", [str(script)])
    runpy.run_path(str(script), run_name="__main__")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.parent.name)
def test_notebook_runs(script: Path, tmp_path: Path):
    """Execute the code cells in order, from a copy of the example folder, as Jupyter would."""
    _needs(script)
    folder = tmp_path / script.parent.name
    shutil.copytree(script.parent, folder)
    os.chdir(folder)  # restored by the fixture's monkeypatch.chdir
    notebook = json.loads(script.with_suffix(".ipynb").read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}  # no __file__, as in a kernel
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            code = compile("".join(cell["source"]), f"{script.stem}.ipynb[{index}]", "exec")
            exec(code, namespace)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.parent.name)
def test_notebook_matches_script(script: Path):
    notebook = script.with_suffix(".ipynb")
    assert notebook.is_file(), f"run: python scripts/build_example_notebooks.py ({notebook.name})"
    assert notebook.read_text(encoding="utf-8") == BUILDER.render(script), (
        f"{notebook.name} is out of date: run python scripts/build_example_notebooks.py"
    )


def test_converter_cells_and_errors():
    text = "# %% [markdown]\n# # Title\n#\n# text\n\n# %%\nx = 1\n\n\n# %%\ny = x\n"
    cells = BUILDER.notebook_from_script(text)["cells"]
    assert [c["cell_type"] for c in cells] == ["markdown", "code", "code"]
    assert cells[0]["source"] == ["# Title\n", "\n", "text"]
    assert cells[1]["source"] == ["x = 1"]
    with pytest.raises(ValueError, match="before the first"):
        BUILDER.notebook_from_script("x = 1\n# %%\n")
    with pytest.raises(ValueError, match="without"):
        BUILDER.notebook_from_script("# %% [markdown]\nnot a comment\n")


def test_every_example_folder_has_its_script(tmp_path: Path):
    (tmp_path / "01_demo").mkdir()
    with pytest.raises(FileNotFoundError, match=r"demo\.py"):
        BUILDER.example_scripts(tmp_path)
