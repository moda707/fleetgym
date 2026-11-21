# Gate 2 report: FleetGym v0.1

Date: 2026-11-20. Branch `main`. Status: **awaiting the user's approval.**

The original Gate 2 (`VRPSD-Outsource-v0`, VRP-SDO) was taken out of scope by the user on 2026-11-20:
its decision is a selection on top of routing and needs its own interface; it will come back with its
model and data. Gate 2 is therefore the review of v0.1 as built (milestones M0 to M10b).

## What v0.1 contains

| Area | Delivered | Decisions |
| --- | --- | --- |
| Problem specs | YAML configs with `extends`, overrides, JSON Schema, content hashes, `fleetgym validate/show/schema` | ADRs 0006, 0007 |
| Instances and networks | VRPLIB and Solomon readers, generators, Euclidean and matrix networks, checksummed downloads | ADR 0008 |
| Simulator | discrete-event engine, exact snapshots, event log, masks from known information | ADR 0009 |
| Validation | independent validator, replication of published solutions | ADRs 0010, 0011 |
| Uncertainty | 13 distributions, revelation, scenarios (Parquet), non-anticipative forks | ADR 0012 |
| Recourse | return to depot, outsource, agent decides, custom rules | ADR 0013, `docs/recourse.md` |
| Dynamic requests | Poisson and NHPP arrivals, cancellations, accept/reject and assign with cheapest insertion, `next_node` with wait, reload and terminate | ADRs 0014, 0015 |
| Multi-agent | PettingZoo AEC and Parallel, conflict rules, reward modes, global state | ADR 0016 |
| Baselines and evaluation | random, nearest, insertion, a priori, rollout, PyVRP re-optimization, perfect information; fixed test sets, CIs, CVaR, paired tests, report cards | ADR 0017 |
| Training | graph encoders (MLP, GAT, Attention Model), pointer decoder, `MaskableGraphPolicy`, `MaskableDQN`, `check_module`, `fleetgym train/eval` | ADR 0018 |
| Visualization | map and Gantt views, `rgb_array`/`human`, `fleetgym render` | ADR 0019 |
| Documentation | Read the Docs site, *Build your own VRP variant* guide, 8 tested examples | ADR 0020 |

Environments: `CVRP-v0` and `VRPTW-v0` (stable), `VRPSD-v0`, `DVRP-StochReq-v0`, `MA-CVRP-v0`,
`MA-DVRP-v0` (experimental); `VRPSD-Outsource-v0` and `EVRP-NL-v0` (planned).

## Evidence

| Claim | Evidence | Result |
| --- | --- | --- |
| Benchmarks reproduce | All 100 CVRPLIB X best-known solutions exactly; all 56 Solomon (SINTEF) to their two decimals and vehicle counts | Pass |
| Masks are exact | Independent oracle (static masks; accept masks of planned routing) | Pass |
| Episodes are valid | 10,000 validated random rollouts (Gate 1); property tests on random static, stochastic and dynamic problems (thousands of examples); every evaluated episode validated | Pass |
| Expected costs are exact | 1,024 scenarios, 6 recourse configurations, equal to an independent dynamic program to 1e-9; Poisson Monte Carlo within its 99.9% interval | Pass |
| No information leaks | Fork-based checks of observations, masks and policy decisions on every stochastic and dynamic setting; the check flags the anticipative oracle | Pass |
| Baselines are sound | Exact hindsight cost bounds every non-anticipative baseline on tiny instances; rollout equals the hand-computed improvement step | Pass |
| Agents learn | MaskablePPO reaches the optimal route of a fixed tiny instance (slow test) | Pass |
| Interoperability | Gymnasium `check_env` on every single-agent config; PettingZoo API tests on both multi-agent configs | Pass |
| Determinism | Identical event logs for the same spec and seed, across processes and in CI on Linux, macOS and Windows | Pass |

Test counts at this gate: 723 tests in the default run (with the optional extras), 682 without them
(core coverage 95%), 165 in the gate suite. CI: lint and type check, 9 test jobs (3 OS x Python 3.12
to 3.14), extras, documentation: all green.

## Known limitations and backlog

- Not in v0.1 by design: VRP-SDO (deferred by the user), EV module, time-dependent travel, GPU backend,
  C++ core, OSM networks, web replay viewer, leaderboard.
- Recourse backlog (`docs/recourse.md`): preventive restocking policies, time-window and battery
  failures, duration limits with recourse, multiple depots, stochastic customers, validator hooks for
  custom rules.
- Open item: the timing of cancellations has no reference model (uniform between release and
  deadline for now; user, 2026-11-20).
- Uncertain parameters together with dynamic requests, and recourse with time windows, raise
  `NotInScopeError`.
- Default hyperparameters are untuned starting points; no published learning results yet.
- Released as `fleetgym 0.1.0a1` on PyPI (2026-11-20).

## For the user

1. Approve Gate 2, or list what must change first.
2. Release: `0.1.0a1` is published; decide when to release `v0.1.0`.
