# 0002: Configs live inside the package

Status: accepted (2026-11-20)

## Context
Design §5.1 shows a top-level `configs/` folder scanned to build the registry. A top-level folder is not
part of the installed wheel, so `fg.make("CVRP-v0")` would fail after `pip install fleetgym`.

## Decision
Registered configs live in `src/fleetgym/configs/<family>/<ID>.yaml` and ship as package data. The
registry scans this folder and every folder registered through the `fleetgym.configs` entry-point group.
A config's file stem must equal its `id`; ID collisions across folders raise an error naming both files.

## Alternatives
- Top-level `configs/` copied into the wheel by a build hook: two locations for one thing, easy to break.
- Configs embedded as Python dicts: loses the "one YAML file per variant" property.

## Consequences
Users editing a shipped config should copy it and use `extends:` or a path, not edit site-packages.
