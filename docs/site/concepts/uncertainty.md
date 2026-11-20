# Uncertainty, scenarios and forks

Every uncertain parameter is declared with a distribution and a revelation time:

```yaml
uncertainty:
  demand: {dist: poisson, mean: from_instance, revealed: on_arrival}
  service_time: {dist: gamma, shape: 2.0, mean: 5.0, revealed: on_completion}
```

## Distributions

`fixed`, `uniform`, `normal`, `truncated_normal`, `lognormal`, `gamma`, `weibull`, `exponential`,
`poisson`, `bernoulli`, `categorical`, `empirical` and `kde`. Each states its parameterization in its
docstring, offers `mean` and `quantile`, and a finite `support` when it has one. Parameters can be
per node (`mean: from_instance`). Add your own with `fleetgym.register_distribution`
({doc}`../guides/build-your-own-variant`).

## Revelation

Demand: `a_priori`, `on_request`, `on_departure`, `on_arrival`; service time also `on_completion`.
Before that, the agent sees the forecast (the mean, or 0 with `forecast: none`).

## Scenarios

All random values of an episode are sampled once, at reset, into a `Scenario`: one value per node and
source, plus the requests of the day with dynamics. Each source draws from its own random stream
keyed by its name, so changing one source never changes the others. Scenarios are saved as Parquet
and replayed exactly with `env.reset(options={"scenario": scenario})`, which gives every policy the
same days (common random numbers).

## Forks

`env.fork(seed)` copies the environment, keeps every revealed value, and redraws everything still
hidden: unrevealed demands, requests that have not arrived, future cancellations (conditioned on not
having happened yet). Rollout and scenario-based planners must use forks; `clone()` would let them see
the future. `fleetgym.testing.check_non_anticipativity` verifies, at every epoch, that observations,
masks and info are the same on forks, and `check_policy_non_anticipative` does the same for a
policy's decisions.
