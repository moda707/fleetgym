# 0011: Solomon replication source and tolerance

Status: accepted (2026-11-20). Resolves plan Q18 and Q19 for the Solomon family.

## Context
`VRPTW-v0` needs a model-fidelity check against published Solomon best-known solutions. Published
values differ by source: heuristic tables use a hierarchical objective (vehicles, then distance)
with double-precision distances, while exact-method papers often minimize distance alone with
distances truncated to one decimal. The two are not comparable, and the user asked whether the
solutions could be found online rather than supplied.

## Decision
- Source: SINTEF TOP, `https://www.sintef.no/projectweb/top/vrptw/100-customers/`. It publishes, for
  all 56 instances with 100 customers, the vehicles, the distance rounded to two decimals, the
  reference, and a route file per instance. Instances come from the `solomon-100.zip` archive linked
  on the same page.
- The page states its conventions: "hierarchical objective: 1) Minimize number of vehicles
  2) Minimize total distance", "the value of travel time is equal to the value of distance", and
  "Distance and time should be calculated with double precision, total distance results are rounded
  to two decimals". A footnote says that earlier values for nine instances came from lower-precision
  arithmetic and were corrected. This is `VRPTW-v0` as written (`rounding: none`, lexicographic
  `[vehicles, distance]`, hard windows, waiting allowed, route duration up to the depot's due time).
- Tolerance (Q19): a replay passes if it is feasible, the independent validator accepts it, the
  vehicle count equals the published one, and the recomputed distance rounds to the published value
  (absolute difference at most 0.005). The manifest records `tolerance: 0.005`; CVRPLIB X keeps 0.
- Files are downloaded on demand and checked against SHA-256 checksums in
  `fleetgym/data/manifests/solomon_100.json` (built by `scripts/build_solomon_manifest.py`). The
  downloader verifies the archive and then the extracted member. Nothing is redistributed: the page
  states no license.
- The table labels two rows `r205b` and `rc101b`; their files are `r205.txt` and `rc101.txt`, and the
  manifest names instances after the files (the label is kept as `table_label`). Table comments are
  kept as `note`.
- With fidelity shown on both benchmark families, `CVRP-v0` and `VRPTW-v0` become `stable` (defined in
  `Meta.status` as "passes the replication suite"). `meta` is excluded from the content hash, so
  their hashes do not change.

## Result (2026-11-20)
All 56 solutions replay feasibly, validate, use the published number of vehicles, and round to the
published distance; the largest absolute difference is 0.00497.

## Alternatives
- Distance-only optima with one-decimal truncation: would need a `truncate_1dp` rounding mode and a
  separate config, because `VRPTW-v0` is hierarchical. Can be added later as its own collection.
- Gehring and Homberger instances (200 to 1000 customers): same source and conventions; deferred to
  keep the gate suite fast.

## Consequences
- `fleetgym replicate VRPTW-v0` works (instances are read with the collection's format, and the
  published cost comes from the manifest when the solution file has none).
- If SINTEF updates a file, the checksum check fails and the manifest must be rebuilt after review.
