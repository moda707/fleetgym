# Installation

FleetGym needs Python 3.12 or later.

```bash
pip install fleetgym            # environments, baselines, evaluation
pip install "fleetgym[sb3]"     # + training: PyTorch, Stable-Baselines3, sb3-contrib
pip install "fleetgym[solvers]" # + PyVRP re-optimization baseline
pip install "fleetgym[viz]"     # + figures (matplotlib)
pip install "fleetgym[all]"     # everything
```

FleetGym is in pre-release: `pip install --pre fleetgym` picks the latest development version.

## From source

```bash
git clone https://github.com/moda707/fleetgym.git
cd fleetgym
uv sync --all-extras
uv run pytest
```

## Benchmark data

Benchmark instances and best-known solutions (CVRPLIB, Solomon) are not part of the package. They
are downloaded when needed, checked against SHA-256 checksums, and cached under
`$FLEETGYM_DATA_DIR` (default `~/.cache/fleetgym`).
