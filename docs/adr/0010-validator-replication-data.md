# 0010: Independent validator, replication and benchmark data

Status: accepted (2026-11-20)

## Validator
`fleetgym.testing.validator.validate(log, spec, instance)` re-checks an episode from the event log, the
spec and the instance. It imports only layers L0 to L2 (enforced by the import-layer test) and
reimplements distances, travel times, service starts, loads and costs from the rules in ADR 0009.
It reports every violation with a code and the log row. The duplication is deliberate: a simulator
bug cannot hide itself. Validator correctness is tested with one corruption test per violation code.

The lexicographic weight base moved from the interface layer to `ObjectiveSpec.scalar_weights()` in
the spec layer, because it is part of what the objective means and the validator needs it.

## Gate 1 evidence
1. **Mask exactness (both directions)**: an independent oracle in `tests/gate1/oracle.py` decides,
   for every pending customer at every epoch, whether "go there now, then return to the depot" is
   feasible. The mask must equal the oracle exactly, so masks are neither too loose nor too strict.
2. **Validator agreement**: 10,000 random masked rollouts across the registered configs and random
   Hypothesis problems; the validator accepts every one (`tests/gate1`, marker `slow`).
3. **Golden optima**: exhaustive search through the environment (snapshot/restore) equals a
   brute-force partition solver on instances with up to 5 customers, for distance, distance plus
   vehicles, and lexicographic `[vehicles, distance]` with time windows.
4. **Model fidelity**: published best-known solutions are replayed as action sequences; they must be
   feasible, valid, and reproduce the published cost.

## Benchmark data
- CVRPLIB files are downloaded on demand from `https://galgos.inf.puc-rio.br/cvrplib/` (the old
  `vrp.galgos...` paths redirect incorrectly) and cached under `$FLEETGYM_DATA_DIR` or
  `~/.cache/fleetgym`. They are not redistributed in the repository.
- Each collection has a manifest in `fleetgym/data/manifests/` with download paths, SHA-256 checksums
  and published costs, built by `scripts/build_cvrplib_manifest.py`. Checksums record the files as of
  the build date; a mismatch is an error (best-known solutions improve upstream, and the change should
  be reviewed, not silently accepted).
- CVRPLIB states the conventions of set X on its site: distances are Euclidean rounded to the nearest
  integer (TSPLIB), cost is total distance, the number of routes is free. This is `CVRP-v0`.
- In CVRPLIB solution files, customer `k` is instance node `k` (depot = node 0); confirmed by replay
  against the published costs of all 100 X instances.
- `fleetgym replicate <ID>` replays a config's collection (`meta.instances` names it).

## Alternatives
- Bundling instance files: avoided for licensing reasons (plan Q18).
- Downloading the per-set 7z archives: needs an extra dependency (py7zr); per-file downloads do not.
