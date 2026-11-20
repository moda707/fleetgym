# 0012: Uncertainty, scenarios and non-anticipativity

Status: accepted (2026-11-20). Adopts plan Q21, Q22 and Q23 (recommendations; user available but
these were not raised as needing input).

## Context
M5 makes the spec's `uncertainty` section run: samplers for the declared distributions, a `Scenario`
of pre-sampled exogenous values (architecture rule 6), revelation to the agent, and planning copies
that do not see the future (rule 10).

## Decisions
- **Samplers** (`fleetgym.uncertainty.distributions`): one class per spec distribution, with the
  parameterization written in its spec model. Parameters broadcast per node (for example a Poisson
  mean per customer from `mean: from_instance`). Each offers `sample`, `mean`, `quantile` and a
  finite `support` (or `None`). `KDE` has no quantile (raises). Custom distributions register a spec
  model and a class with `register_distribution`.
- **Sources in this version**: `demand` and `service_time`. `presence` (stochastic customers) is now
  a reserved source (after v0.1, recourse backlog item 8). Stochastic travel times stay deferred
  (Q21).
- **Revelation**: demand `a_priori`, `on_request`, `on_departure` or `on_arrival`; service time also
  `on_completion`. `on_request` equals `a_priori` for customers known at the start. `never` is
  rejected (the vehicle learns the value while serving). Unrevealed values appear to the agent as the
  forecast: the distribution's mean (`forecast: mean`) or 0 (`forecast: none`); the observation has a
  demand-known flag per node (node features grow from 8 to 9).
- **Scenario**: one realized value per node and source; depot values are deterministic (demand 0,
  the instance's service time); negative or non-finite customer values raise a `ConfigError` at the
  source's key path (no clipping). Each source draws from
  `StreamFactory(seed).generator("uncertainty.<source>")`, so adding a source does not change another
  source's draws. Parquet I/O keeps values bit for bit; the scenario id is a content hash of the
  spec hash, instance hash and values. `reset(options={"scenario": s})` replays a scenario; a
  scenario for another spec or instance raises `ScenarioMismatchError`.
- **Masks under uncertainty**: the capacity constraint uses known demands only; a customer whose
  demand is not revealed is feasible, and a capacity failure is handled by recourse (ADR 0013). A
  spec with late-revealed demand and a capacity constraint must name `recourse.capacity_failure`.
- **Scope limits**: recourse together with time windows, route duration or shift limits, and
  stochastic service times together with hard time windows, route duration or shift limits, raise
  `NotInScopeError` (they need time-window recourse, backlog items 3 and 6).
- **clone vs fork (Q22)**: `clone()` copies everything, including the true future. `fork(seed)`
  keeps the state and every revealed value and redraws everything still hidden. Planners (rollout,
  scenario methods) must use `fork`. A service already in progress keeps its scheduled end time in a
  fork (a small, documented approximation for service times revealed on completion).
- **Non-anticipativity checker**: at every epoch, forks must produce the same observation, mask and
  agent-facing info. The scenario id is therefore not in `info` (it would identify the future); it
  stays in the log metadata and on `env.scenario`.
- **Determinism (Q23)**: exact on the same platform; across platforms, discrete outcomes exactly and
  floats to 1e-9 relative.
- **Validator**: takes the scenario and checks revealed values against it.

## Alternatives
- Drawing per customer from separate streams: unnecessary for static customers; per-source streams
  already isolate sources. Revisit for dynamic requests (M6).
- Clipping negative draws to 0: a silent change of the distribution; rejected.

## Consequences
- Observation layouts changed for all environments (one more node feature). Spec hashes of existing
  configs are unchanged.
- Event log schema version 2 adds `reveal`, `failure`, `restock` and `outsource` rows.
