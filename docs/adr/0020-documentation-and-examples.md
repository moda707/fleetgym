# 0020: Documentation site and examples

Status: accepted (2026-11-20). Requested by the user (plan M10b); dependencies approved by the user.

## Decision
- Sphinx site in `docs/site/` with the `pydata-sphinx-theme` (the common choice of scientific Python
  projects), pages in Markdown (MyST), NumPy docstrings (napoleon) and type hints
  (`sphinx-autodoc-typehints`); extra `docs`. Read the Docs builds it from `.readthedocs.yaml` (the
  project must be imported once on readthedocs.org by the repository owner).
- Structure: getting started, concepts (specs, simulation, uncertainty, recourse, dynamic requests,
  multi-agent, validation), environment catalog, guides (build your own variant, stochastic and
  dynamic problems, baselines and evaluation, training, visualization), reference (API, CLI, design
  decisions, changelog).
- The guide **Build your own VRP variant** is the centerpiece: YAML only, then a custom distribution,
  constraint and recourse rule, then a plugin package. `fleetgym.register_constraint` was added so
  constraints extend like distributions and recourse rules.
- Examples in `examples/` are the single source of code for the guides (`literalinclude`), and the
  test suite runs every example; a test checks that every included file exists. The docs build with
  `-W` in CI and in a slow test.
- Each example lives in its own folder (`examples/NN_topic/topic.py`) with a notebook next to it
  (user, 2026-11-20). The script is the source of truth, split into cells with `# %%` markers;
  `scripts/build_example_notebooks.py` generates the notebook (nbformat 4, no outputs) without new
  dependencies. Tests run every script, execute every notebook's code cells in order from a copy of
  its folder (no `__file__`, as in a kernel), and fail when a notebook is out of date. Generated
  notebooks are excluded from ruff. Alternatives: jupytext (a new dependency) or hand-maintained
  notebooks (drift from the scripts the docs include).

## Consequences
- Documentation code cannot silently go stale; examples that need an extra are skipped without it.
- A gallery (`sphinx-gallery`) and API pages for every module can come later.
