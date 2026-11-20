# 0017: Baselines and evaluation

Status: accepted (2026-11-20). Adopts plan Q29 (recommendation) and reports on Q30.

## Policy protocol
`reset(env)` binds a policy to an environment (at the start or at any state, for rollouts);
`act(obs, info) -> action` decides. `anticipative` marks policies that read the true future.
`run_episode` plays one episode and records costs and decision times.

## Baselines
- `RandomPolicy` (own seeded generator), `NearestNeighbor`, `CheapestInsertion` (repeated cheapest
  insertion of the deciding vehicle's route with `sim/routing_rules.py`, Q27), `AprioriRoute`.
  Request epochs (`accept_reject`, `assign`) accept when feasible, with the cheapest vehicle.
- `Rollout`: one-step lookahead on forks (`FleetGymEnv.fork`), base policy to the end, mean over
  `n_samples`; on deterministic configs it is the exact improvement step of the base policy.
- `PyVRPReoptimization` (extra `solvers`): re-solves the known problem at every vehicle epoch.
  **Q30 findings** (PyVRP 0.14.1, checked against its installed API): vehicles on the road are
  modeled natively (per-type start depot at the vehicle's next free location, `tw_early`, remaining
  capacity, remaining `shift_duration`); optional clients take a prize. Two adaptations were needed:
  (1) integers: distances and times scaled by 1000 unless already integer; (2) PyVRP charges nothing
  for an unused vehicle, but a vehicle on the road must still drive home, so each one gets a required
  dummy client at its position that only it can reach (its own routing profile). Soft windows,
  recourse and waiting rules are not modeled; a suggestion that is not valid falls back to nearest
  neighbor and is counted (`fallbacks`). Route activities report clients through `is_client()` (a
  method) and the client index `idx`.
- `PerfectInformation` (anticipative): `exact` searches every action sequence on clones and is a
  **lower bound** on non-anticipative policies for that scenario (tiny instances; Q29); `rollout` is a
  clone-based lookahead, a **perfect-information heuristic**, not a bound. Report cards must say
  which (`report_card(..., bound=...)` is required for anticipative results).

## Evaluation
- `scenario_seeds(env_id, split, n)`: seeds derived from the ID and split; train, validation and test
  never overlap.
- `evaluate` runs a policy on seeds and validates every episode with the independent validator
  (`eval` may import `testing`, an allowed same-layer import).
- Metrics: mean with a Student-t 95% CI, CVaR at 90% (mean of the worst `ceil(10% n)` episodes),
  mean cost components, decision times. `compare`: paired t-test and Wilcoxon (scipy).
- `report_card`: spec hash, seeds, costs, summary, package versions, platform, training settings,
  optional paired comparison and gap to a reference.

## Verification
Every baseline gives validator-accepted episodes on every single-agent config it supports; policy
non-anticipativity (decisions unchanged on forks) for nearest neighbor, insertion and rollout, and
the check flags the oracle; the exact hindsight cost is at most every non-anticipative policy's cost
on tiny deterministic and stochastic instances; rollout equals the hand-computed improvement step;
PyVRP beats nearest neighbor on the configs; statistics agree with scipy; baseline costs on fixed
seeds are snapshotted (`tests/golden/baseline_costs.json`).

## Not in this milestone
Multiple scenario approach, sample scenario hedging, waiting strategies and expected-value planning
from design §18 are after v0.1; decision-time budgets are measured but not enforced.
