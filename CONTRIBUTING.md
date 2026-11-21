# Contributing to FleetGym

Thanks for helping. FleetGym aims to be a library whose results people can trust in papers, so
correctness and reproducibility come before speed and features.

## Setup

```bash
git clone https://github.com/moda707/fleetgym.git
cd fleetgym
uv sync --all-extras
uv run pre-commit install
```

## Checks (all must pass)

```bash
uv run pytest                                   # default suite (no slow or network tests)
uv run pytest -m "slow or network" tests/gate1  # benchmark replication and the 10,000-rollout gate
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

Reproduce the default CI test job (no optional extras) before pushing changes that touch optional
modules, since a module that imports an extra must not break installs without it:

```bash
UV_PROJECT_ENVIRONMENT=.venv-core uv sync --extra dev
UV_PROJECT_ENVIRONMENT=.venv-core uv run --no-sync pytest -n auto --cov --cov-config=.github/coverage-core.toml
```

After editing an example script, regenerate its notebook with
`uv run python scripts/build_example_notebooks.py` (see `examples/README.md`).

Build the documentation with `uv sync --extra docs --extra viz` and
`uv run sphinx-build -W -b html docs/site docs/_build/html`.

## Ground rules
- Read [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture and [`docs/adr/`](docs/adr/) for
  decisions already made. To change a decision, propose a new ADR instead of editing code around it.
- Modules import only from lower layers (enforced by `tests/architecture/test_layers.py`). The
  validator in `fleetgym.testing.validator` stays independent of the simulator.
- Write tests with the change: unit tests, determinism (same spec and seed give the same
  trajectory), Hypothesis property tests where inputs vary, and golden tests on tiny instances.
- Changing an environment's dynamics or defaults changes its spec hash: bump its `-vN` and update
  `tests/golden/`.
- Benchmark numbers (published costs, parameters, conventions) come from the original files or
  papers, never from memory. Cite the source in the manifest or ADR.
- No silent fallbacks: raise a `FleetGymError` subclass. Config errors name the file and the dotted
  key path.
- NumPy-style docstrings on public objects; American English.
- New runtime dependencies need discussion in an issue first.

## Adding an environment
Add one YAML file under `src/fleetgym/configs/<family>/<ID>.yaml` (the file stem is the ID), usually
extending an existing config. Start with `meta.status: experimental`; it becomes `stable` once a
replication test reproduces published results.

## Releasing (maintainers)
Releases go to PyPI through trusted publishing (no API token is stored anywhere).

One-time setup:
1. Create an account on <https://pypi.org> and enable two-factor authentication.
2. In *Your account → Publishing → Add a new pending publisher*, choose GitHub and enter:
   PyPI project name `fleetgym`, owner `moda707`, repository `fleetgym`, workflow `release.yml`,
   environment `pypi`.
3. In the GitHub repository, *Settings → Environments → New environment* named `pypi` (optionally
   with required reviewers, so every upload needs a click).

Each release:
1. Set `version` in `pyproject.toml` and `CITATION.cff`, update `CHANGELOG.md`, commit.
2. Create a GitHub release with tag `v<version>` (for example `v0.1.0a1`). The `release`
   workflow checks that the tag matches the version, builds, smoke-tests the wheel and uploads it.

A pending publisher does not reserve the name on PyPI: the name is taken by the first upload.
