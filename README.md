# FleetGym

Gymnasium and PettingZoo environments for reinforcement learning on vehicle routing problems,
with a focus on stochastic and dynamic variants where information arrives during the day.

> Status: pre-alpha. The v0.1 core is under construction; see `STATUS.md` and
> `docs/IMPLEMENTATION_PLAN.md`. APIs will change.

## Install (development)

```bash
uv sync --extra dev
uv run pytest
```

Requires Python 3.12 or later. Optional extras: `sb3`, `gnn`, `solvers`, `viz`, `all`.

## Citing

If you use FleetGym in your research, please cite it. GitHub's "Cite this repository" button and
`CITATION.cff` give the citation; a Zenodo DOI will be added with each release.

## License

MIT. See `LICENSE`.
