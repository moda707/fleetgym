**What and why**

**Checklist**
- [ ] Tests added with the change (unit, determinism, property or golden as relevant)
- [ ] `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` pass
- [ ] An environment's dynamics or defaults changed: its `-vN` is bumped and `tests/golden/` updated
- [ ] Non-obvious decision recorded in `docs/adr/`; `CHANGELOG.md` updated
