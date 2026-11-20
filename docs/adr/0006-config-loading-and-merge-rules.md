# 0006: Config loading, merge rules and overrides

Status: accepted (2026-11-20). Resolves plan Q7 (user-approved) and Q11 (recommendation adopted).

## Context
Design §5.1 defines `extends` with deep merge, merge-by-name for constraints and `null` deletion.
Applied literally, the design's own `DVRP-StochReq-v0` (extends `VRPTW-v0`) would end up with both a
lexicographic and a weighted objective, and every child would inherit its parent's `meta.expected`
replication targets. The design proposes Hydra for overrides, which is not an allowed dependency.

## Decision
Pipeline: read YAML → resolve `extends` (ID via the registry, or a path relative to the file) →
merge → apply overrides → validate. Rules:

- Maps merge deeply; scalars and lists replace. `constraints` is a map keyed by name, so it merges
  by name.
- `null` removes the key, so the default applies. This holds everywhere, including in a file without
  `extends`: `null` always means "use the default".
- A child map that changes a discriminator (`dist`, `type`, `process`, `model`, `function`, `rule`)
  replaces the parent's map instead of merging into it.
- `objective` is always replaced (user decision, Q7).
- `meta` and `id` are never inherited (user decision, Q7).
- Overrides use a small built-in parser: `a.b.0.c=value` with YAML-typed values, `null` deletes,
  intermediate maps are created, list indices must exist, and an override replaces (never merges).
  `load_spec(source, overrides=[...])`; the CLI uses `--set KEY=VALUE`.
- YAML: safe loader, duplicate keys are errors, and floats follow YAML 1.2 (`1e-6`, `1.5e2` are floats).
- Errors: every validation problem is reported in one `ConfigError`, each with the file (or override)
  that last set the value and the dotted key path in the YAML.

## Alternatives
- Pure deep merge everywhere (as written in the design): produces invalid specs for the design's own
  examples.
- Hydra/OmegaConf: a large dependency for one feature, with its own interpolation semantics.

## Consequences
`fleetgym show <id>` prints the merged file; `--defaults` prints the validated spec with every default.
