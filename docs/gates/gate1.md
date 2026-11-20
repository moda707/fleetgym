# Gate 1 report

Date: 2026-11-20. Branch `v0.1`. Status: **awaiting approval**; one item pending user input.

| Requirement (build request M4) | Evidence | Result |
| --- | --- | --- |
| Independent validator | `fleetgym.testing.validator`, imports only L0 to L2 (layer test); 13 corruption tests, one per violation code | Pass |
| Masks and validator agree on 10,000 random rollouts | `tests/gate1/test_masks_and_validator.py::test_gate1_ten_thousand_rollouts` (4,000 rollouts of `CVRP-v0`/`VRPTW-v0`, 6,000 random Hypothesis problems with up to 8 customers, all epoch, fleet and constraint combinations) | Pass |
| Masks are exact (not too loose, not too strict) | Independent oracle: at every epoch a customer is unmasked iff "go there now, then return" is feasible | Pass |
| Model fidelity: CVRPLIB | All 100 X-set best-known solutions (CVRPLIB, retrieved 2026-11-20, SHA-256 checked) replay feasibly, validate, and reproduce the published cost exactly (integer costs, tolerance 0) | Pass |
| Model fidelity: Solomon | Needs the user's choice of BKS files and distance convention (plan Q18/Q19) | **Pending user input** |
| Golden tests with brute-force optima | Exhaustive search through the environment equals a brute-force partition solver (CVRP: distance, distance + vehicles; VRPTW: lexicographic vehicles then distance) | Pass |
| `check_env` for every registered config | Gymnasium `check_env` plus validator agreement, determinism, clone exactness (`fleetgym.testing.env_checker`) | Pass |

## How to rerun

```bash
uv run pytest -m "slow or network" tests/gate1 -n auto   # full gate (downloads CVRPLIB X, ~25 MB)
uv run fleetgym replicate CVRP-v0                         # fidelity table
```

## Open item for the user (Solomon)
Solomon best-known solutions are published with different conventions (double-precision distances
vs. truncation to one decimal; vehicles-first hierarchical objective vs. distance only). Please say
which source and convention `VRPTW-v0` should reproduce, ideally with solution files. The test
`tests/gate1/test_fidelity.py::test_solomon_fidelity` is skipped until then.
