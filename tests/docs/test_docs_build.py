"""The documentation builds without warnings (extra ``docs``)."""

from __future__ import annotations

from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parents[2] / "docs" / "site"


@pytest.mark.slow
@pytest.mark.filterwarnings("default")  # library deprecations; Sphinx's -W covers the docs
def test_docs_build_without_warnings(tmp_path: Path):
    pytest.importorskip("sphinx")
    pytest.importorskip("pydata_sphinx_theme")
    from sphinx.cmd.build import build_main

    assert build_main(["-W", "--keep-going", "-q", "-b", "html", str(SITE), str(tmp_path)]) == 0
    assert (tmp_path / "guides" / "build-your-own-variant.html").exists()


def test_docs_include_only_existing_examples():
    """Every literalinclude points at a file that exists (the examples are tested)."""
    for page in SITE.rglob("*.md"):
        for line in page.read_text(encoding="utf-8").splitlines():
            if line.startswith("```{literalinclude}"):
                target = (page.parent / line.split("}", 1)[1].strip()).resolve()
                assert target.exists(), f"{page.name}: {target}"
