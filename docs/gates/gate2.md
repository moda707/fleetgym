# Gate 2 report: FleetGym v0.1

Date: 2026-11-21 (first version 2026-11-20). Branch `main` at `6c41e80`. Status: **approved by the user (2026-11-21)**;
released as v0.1.0.

The original Gate 2 (`VRPSD-Outsource`, VRP-SDO) was taken out of scope by the user on 2026-11-20:
its decision is a selection on top of routing and needs its own interface; it will come back with its
model and data. Gate 2 is therefore the review of v0.1 as built: milestones M0 to M10b, plus the
extensions the user added on 2026-11-20 and 2026-11-21 (M12 to M15).

## What v0.1 contains

| Area | Delivered | Decisions |
| --- | --- | --- |
| Models | YAML specs with `extends`, overrides, JSON Schema, content hashes; families `CVRP-v1`, `VRPTW-v1`, `SVRP-v1`, `DVRP-v1`, `EVSPP-v1`; `fleetgym validate/show/schema` | ADRs 0006, 0007, 0023 |
| Instances | One JSON format (`fleetgym-instance/1`) with extensible schemas (`fleetgym/core/1`, `fleetgym/evspp/1`, user schemas in YAML); instance sets with checksummed manifests; CVRPLIB and Solomon readers and checksummed downloads. No instances or generators ship: every number comes from instances | ADRs 0008, 0023 |
| Paper presets | 10 published settings with their grids: Solomon, Florio et al., Kool et al., Dastpak et al. (VRP-VCSD, Goodson's VRPSD, EVSPP-OI), Vera and Abad, Ulmer et al., Taş et al., Errico et al. | ADRs 0022 to 0026 |
| Simulator | Discrete-event engine, exact snapshots, event log, masks from known information; multi-trip routes, reload times, soft and hard duration limits, end-of-route rules | ADRs 0009, 0021, 0023 |
| Validation | Independent validators (routing; EV shortest path), replication of published solutions | ADRs 0010, 0011, 0026 |
| Uncertainty | 18 distributions; uncertain demands, service times and travel times (per arc); revelation; scenarios (Parquet); non-anticipative forks | ADRs 0012, 0025 |
| Recourse | Return to depot, outsource, agent decides, split (partial service with duration limits), custom rules; grammar `{serve, remainder}` | ADRs 0013, 0021, 0023 |
| Dynamic requests | Poisson and NHPP arrivals, clustered locations, cancellations; accept/reject and assign with cheapest insertion, decided at release or at vehicle arrivals; mandatory early requests; `next_node` with wait | ADRs 0014, 0015, 0024 |
| Time windows | Hard and soft, waiting or service on arrival (earliness), depot window enforced or ignored | ADRs 0023, 0025 |
| EV shortest path | One EV over charging stations with M/M/1/kappa queues and real-time occupancy indicators; diversion while driving; its own environment, simulator and validator | ADR 0026 |
| Multi-agent | PettingZoo AEC and Parallel, conflict rules, reward modes, global state | ADR 0016 |
| Baselines and evaluation | Random, nearest, insertion, a priori, rollout, PyVRP re-optimization, perfect information, EV greedy and steady-state re-optimization; fixed test sets, CIs, CVaR, paired tests, report cards | ADRs 0017, 0026 |
| Training | Graph encoders (MLP, GAT, Attention Model), pointer decoder, `MaskableGraphPolicy`, `MaskableDQN`, `check_module`, `fleetgym train/eval --instances` | ADR 0018 |
| Visualization | Map and Gantt views, `rgb_array`/`human`, `fleetgym render` (routing environments) | ADR 0019 |
| Documentation | Read the Docs site (concepts, guides, API), *Build your own VRP variant* with a custom instance schema, 18 tested examples (8 tutorials, 10 paper settings), 26 ADRs, 3 design proposals | ADR 0020 |

Model families: `CVRP-v1`, `VRPTW-v1` (stable); `SVRP-v1`, `DVRP-v1`, `EVSPP-v1` (experimental);
`VRPSD-Outsource-v1` and `EVRP-NL-v1` (planned). Every family can be multi-agent
(`agents: {mode: multi}`) except `EVSPP-v1`.

## Evidence

| Claim | Evidence | Result |
| --- | --- | --- |
| Benchmarks reproduce | All 100 CVRPLIB X best-known solutions exactly; all 56 Solomon (SINTEF) to their two decimals and vehicle counts (through the `-v1` families) | Pass |
| Masks are exact | Independent oracle (static masks; accept masks of planned routing) | Pass |
| Episodes are valid | 10,000 validated random rollouts (Gate 1); Hypothesis property tests on random static, stochastic, dynamic, stochastic-time and EV problems; every evaluated and example episode validated | Pass |
| Expected costs are exact | 1,024 scenarios, 6 recourse configurations, equal to an independent dynamic program to 1e-9; Poisson Monte Carlo within its 99.9% interval | Pass |
| Success probabilities are exact | Simulated success rate of a fixed route under stochastic service times and hard windows matches Errico et al.'s exact recursion (eq. 18-20) within 4 standard errors over 3,000 episodes | Pass |
| No information leaks | Fork-based checks of observations, masks and policy decisions on every stochastic and dynamic setting, including stochastic travel times (in-transit arcs redrawn given the elapsed time) and EV occupancy forks; the check flags the anticipative oracle | Pass |
| Samplers are right | KS and chi-square tests: gamma arc times (Taş eq. 16-17), discrete triangular, normal mixtures, M/M/1/kappa stationary starts, Poisson arrivals, the papers' instance distributions | Pass |
| Paper settings are plausible | Myopic policy on Ulmer et al.'s uniform settings close to their Table 2 (e.g. 25.4 vs 25.3); Errico et al.'s median plans fail often and worst-case plans never, as in their Tables 8-11; EV steady-state baseline drives 8.1 h vs 8:04 in Dastpak et al.'s Table 7 | Indicative |
| Baselines are sound | Exact hindsight cost bounds every non-anticipative baseline on tiny instances; rollout equals the hand-computed improvement step | Pass |
| Agents learn | MaskablePPO reaches the optimal route of a fixed tiny instance (slow test) | Pass |
| Interoperability | Gymnasium `check_env` on every single-agent family (including `EVSPP-v1`); PettingZoo API tests on multi-agent settings | Pass |
| Determinism | Identical event logs for the same spec and seed, across processes and in CI on Linux, macOS and Windows | Pass |

Test counts at this gate: 892 tests in the default run (with the optional extras), 849 without them
(core coverage 93.7%), 159 in the gate suite, 180 that download benchmark data. CI (lint and type
check, 9 test jobs on 3 OS x Python 3.12 to 3.14, extras, documentation) green for every milestone
commit, the last at `6c41e80` (M14).

## Known limitations and backlog

- Not in v0.1 by design: VRP-SDO and the multi-agent DVRP example (deferred by the user), EV routing
  with several customers (`EVRP-NL-v1`), time-dependent travel, GPU backend, C++ core, OSM networks,
  web replay viewer, leaderboard.
- Recourse backlog (`docs/recourse.md`): preventive restocking policies, time-window and battery
  failures, multiple depots, stochastic customers, validator hooks for custom rules.
- Stochastic travel times work with soft time limits only; uncertain parameters together with
  dynamic requests raise `NotInScopeError`.
- EV forks redraw each queue from its occupancy indicator alone, an approximation of the posterior
  (ADR 0026); EV target battery levels are a grid plus the charging breakpoints.
- Open item: the timing of cancellations has no reference model (uniform between release and
  deadline; user, 2026-11-20).
- Choices where papers are silent or inconsistent are recorded in each preset's notes and ADR
  (Errico's impossible skewed medians read as modes; Dastpak et al.'s PM1/PM2 inconsistency settled by
  Section 5.3; Ulmer et al.'s infeasible early customers handled by the 2017 rule).
- Default hyperparameters are untuned starting points; no published learning results yet.
- `fleetgym 0.1.0a1` on PyPI (2026-11-20) predates M12 to M15: the `-v0` IDs and instance generators
  it contains are gone, so its API differs from `main`.

## Decision

The user approved Gate 2 on 2026-11-21 and chose to release `v0.1.0` at once.
