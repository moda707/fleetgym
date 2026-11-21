# Uncertainty, scenarios and forks

Every uncertain parameter is declared with a distribution and a revelation time:

```yaml
uncertainty:
  demand: {dist: poisson, mean: from_instance, revealed: on_arrival}
  service_time: {dist: gamma, shape: 2.0, mean: 5.0, revealed: on_completion}
```

## Distributions

`fixed`, `uniform`, `normal`, `truncated_normal`, `lognormal`, `gamma`, `weibull`, `exponential`,
`poisson`, `bernoulli`, `categorical`, `discrete_uniform` (integers around a per-node center),
`scaled_categorical` (multiples of a per-node scale), `discrete_triangular` (a triangle on a grid),
`empirical`, `kde`, `normal_mixture`, and `gamma_per_distance` for travel times. Each states its
parameterization in its
docstring, offers `mean` and `quantile`, and a finite `support` when it has one. Parameters can be
per node (`mean: from_instance`). Add your own with `fleetgym.register_distribution`
({doc}`../guides/build-your-own-variant`).

## Revelation

Demand: `a_priori`, `on_request`, `on_departure`, `on_arrival`; service time also `on_completion`.
Before that, the agent sees the forecast (the mean, or 0 with `forecast: none`).

## Travel times

`travel_time` is uncertain per arc (Taş et al. 2012):

```yaml
uncertainty:
  travel_time: {dist: gamma_per_distance, shape: 1.0, scale: 1.0, revealed: on_arrival}
```

An arc of length `d` takes Gamma(shape d, scale): the sum of `d` independent unit-distance times,
with mean `shape * scale * d`, which must equal the network's travel time (`shape * scale = 1 /
speed`); masks and observations use these expected times. One time is drawn per arc and episode and
is known when the vehicle arrives. Supported with soft time limits; with hard windows, route
duration or shift limits a vehicle could arrive after a closing time it committed to, which this
version does not model.

## Hard windows and stochastic service times

A vehicle chooses its next stop when its service ends, so it knows when it would arrive: with hard
customer windows, a customer it can no longer serve in time is masked and ends `unserved`. For a
priori routes, a route therefore fails exactly when one of its customers is skipped (the success
probability of Errico et al. 2018). The depot must not close (`time_windows: {depot: ignore}`), and
no hard route duration or shift limit may apply.

## Scenarios

All random values of an episode are sampled once, at reset, into a `Scenario`: one value per node and
source (per arc for travel times), plus the requests of the day with dynamics. Each source draws from its own random stream
keyed by its name, so changing one source never changes the others. Scenarios are saved as Parquet
and replayed exactly with `env.reset(options={"scenario": scenario})`, which gives every policy the
same days (common random numbers).

## Forks

`env.fork(seed)` copies the environment, keeps every revealed value, and redraws everything still
hidden: unrevealed demands and travel times, requests that have not arrived, future cancellations
(conditioned on not having happened yet). A vehicle still on the road keeps its known departure and
gets a travel time drawn given that it has not arrived yet. Rollout and scenario-based planners must use forks; `clone()` would let them see
the future. `fleetgym.testing.check_non_anticipativity` verifies, at every epoch, that observations,
masks and info are the same on forks, and `check_policy_non_anticipative` does the same for a
policy's decisions.
