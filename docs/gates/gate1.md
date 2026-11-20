# Gate 1 report

Date: 2026-11-20. Branch `v0.1`. Status: **approved by the user on 2026-11-20.** The Solomon item,
pending at submission, was closed after approval with public SINTEF data (ADR 0011).

| Requirement (build request M4) | Evidence | Result |
| --- | --- | --- |
| Independent validator | `fleetgym.testing.validator`, imports only L0 to L2 (layer test); 13 corruption tests, one per violation code | Pass |
| Masks and validator agree on 10,000 random rollouts | `tests/gate1/test_masks_and_validator.py::test_gate1_ten_thousand_rollouts` (4,000 rollouts of `CVRP-v0`/`VRPTW-v0`, 6,000 random Hypothesis problems with up to 8 customers, all epoch, fleet and constraint combinations) | Pass |
| Masks are exact (not too loose, not too strict) | Independent oracle: at every epoch a customer is unmasked iff "go there now, then return" is feasible | Pass |
| Model fidelity: CVRPLIB | All 100 X-set best-known solutions (CVRPLIB, retrieved 2026-11-20, SHA-256 checked) replay feasibly, validate, and reproduce the published cost exactly (integer costs, tolerance 0) | Pass |
| Model fidelity: Solomon | All 56 SINTEF best-known solutions (100 customers, retrieved 2026-11-20, SHA-256 checked) replay feasibly, validate, use the published number of vehicles, and round to the published distance (largest difference 0.00497; ADR 0011) | Pass |
| Golden tests with brute-force optima | Exhaustive search through the environment equals a brute-force partition solver (CVRP: distance, distance + vehicles; VRPTW: lexicographic vehicles then distance) | Pass |
| `check_env` for every registered config | Gymnasium `check_env` plus validator agreement, determinism, clone exactness (`fleetgym.testing.env_checker`) | Pass |

## How to rerun

```bash
uv run pytest -m "slow or network" tests/gate1 -n auto   # full gate (downloads CVRPLIB X and Solomon)
uv run fleetgym replicate CVRP-v0                         # fidelity table
uv run fleetgym replicate VRPTW-v0
```

## Solomon (closed after approval)
At submission, the Solomon source and convention were open (plan Q18/Q19). The user asked whether the
solutions could be found online; SINTEF TOP publishes route files and states its convention
(hierarchical objective, double precision, totals rounded to two decimals), which matches
`VRPTW-v0`. See ADR 0011.
