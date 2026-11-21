"""Generate the notebook next to each example script.

Each example is a Python script split into cells with ``# %%`` (code) and ``# %% [markdown]``
(Markdown, every line prefixed with ``# ``) markers. The script is the source of truth; this tool
writes ``<folder>/<script>.ipynb`` (nbformat 4, no outputs) from it.

Usage::

    python scripts/build_example_notebooks.py          # write every notebook (clears outputs)
    python scripts/build_example_notebooks.py --check  # fail if a notebook's cells are out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
MARKER = "# %%"
MARKDOWN = "# %% [markdown]"


def example_scripts(root: Path = EXAMPLES) -> list[Path]:
    """Return the main script of every example folder (the file named like its folder's topic)."""
    scripts = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2].isdigit()):
        main = folder / f"{folder.name[3:]}.py"
        if not main.is_file():
            raise FileNotFoundError(f"{folder}: expected the example script {main.name}")
        scripts.append(main)
    return scripts


def _source(lines: list[str]) -> list[str]:
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    return [line + "\n" for line in lines[:-1]] + lines[-1:]


def notebook_from_script(text: str) -> dict:
    """Convert a cell-marked script into a notebook (a JSON-ready dict)."""
    cells: list[dict] = []
    kind: str | None = None
    lines: list[str] = []

    def flush() -> None:
        if kind is None:
            if any(line.strip() for line in lines):
                raise ValueError("code before the first '# %%' marker")
            return
        if kind == "markdown":
            body = []
            for line in lines:
                if line.strip() and not line.startswith("#"):
                    raise ValueError(f"Markdown cell line without '# ': {line!r}")
                body.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _source(body)})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": _source(list(lines)),
                }
            )

    for line in text.splitlines():
        if line.startswith(MARKER):
            flush()
            kind = "markdown" if line.strip() == MARKDOWN else "code"
            lines = []
        else:
            lines.append(line)
    flush()
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"  # required by nbformat 4.5; stable ids keep files in sync
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render(script: Path) -> str:
    """The notebook text for ``script``, exactly as written to disk."""
    notebook = notebook_from_script(script.read_text(encoding="utf-8"))
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def _cells(text: str) -> list[tuple[str, str, str]]:
    notebook = json.loads(text)
    return [(c["cell_type"], c.get("id", ""), "".join(c["source"])) for c in notebook["cells"]]


def main(argv: list[str] | None = None) -> int:
    """Write or check every example notebook; return the exit code."""
    parser = argparse.ArgumentParser(description="Generate the example notebooks.")
    parser.add_argument("--check", action="store_true", help="only report stale notebooks")
    args = parser.parse_args(argv)
    stale = []
    for script in example_scripts():
        path = script.with_suffix(".ipynb")
        text = render(script)
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            continue
        saved = path.read_text(encoding="utf-8") if path.is_file() else None
        if args.check and saved is not None and _cells(saved) == _cells(text):
            continue  # only outputs of a local run differ
        stale.append(path)
        if not args.check:
            path.write_text(text, encoding="utf-8")
    for path in stale:
        verb = "stale" if args.check else "wrote"
        print(f"{verb}: {path.relative_to(EXAMPLES.parent)}")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
