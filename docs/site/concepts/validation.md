# Validation and reproducibility

FleetGym is meant for papers, so correctness is tested, not assumed.

- **Independent validator** (`fleetgym.testing.validator.validate`): re-checks any episode from its
  event log, the spec, the instance and the scenario, with its own implementation of the rules. It
  imports no simulator code (enforced), so a simulator bug cannot hide itself. The evaluation harness
  validates every episode it reports.
- **Benchmarks**: all 100 CVRPLIB X best-known solutions replay in `CVRP-v1` with their exact published
  cost, and all 56 Solomon best-known solutions (SINTEF) in `VRPTW-v1` to their published two
  decimals and vehicle counts (`fleetgym replicate`).
- **Exactness**: masks equal an independent feasibility oracle; expected costs under recourse equal an
  independent dynamic program; the perfect-information cost bounds every non-anticipative policy on
  tiny instances.
- **Determinism**: the same spec, instance and seed give the identical event log (also across
  processes and operating systems in CI).
- **Property tests**: random specs, instances and action sequences (Hypothesis) never crash, keep the
  simulator's invariants and validate.
