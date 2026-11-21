# ADR 0025: Stochastic travel and service times (M15)

Date: 2026-11-21. Status: accepted (the user approved proposal 0002).

## Context

The user asked for the settings of Taş, Dellaert, van Woensel and de Kok (2012; stochastic travel
times, soft time windows) and Errico, Desaulniers, Gendreau, Rei and Rousseau (2018; hard time
windows, stochastic service times). `travel_time` was a reserved uncertainty source, uncertainty was
per node, and stochastic service times were rejected with hard windows. The design is in
`docs/proposals/0002-stochastic-travel-and-service-times.md`.

## Decision

- `uncertainty.travel_time` with `gamma_per_distance: {shape, scale}`: arc (i, j) takes
  Gamma(shape d_ij, scale). The mean must equal the network's travel time (`shape * scale = 1 /
  speed`), so masks and observations keep using the network's expected times. One draw per arc and
  episode, stored in the scenario as a flattened n x n source (`MATRIX_SOURCES`) and sampled from the
  network's distances (`sample_scenario(..., distances=...)`). Revealed on arrival only.
- The engine schedules the arrival at the realized time but keeps the vehicle's known time at the
  expected arrival, and charges the `time` cost on arrival (a reward at departure would reveal the
  realized time). A fork redraws an in-transit arc given that the vehicle has not arrived:
  `F^-1(F(e) + F(x) (1 - F(e)))`. The validator replays the scenario's arc times.
- Stochastic travel times work with soft time limits only, without dynamic requests or recourse;
  hard windows, hard durations and shifts raise `NotInScopeError`.
- `time_windows.early: serve` (soft windows): service starts on arrival and the time before the
  window opens is charged as `earliness` (new cost component).
- Stochastic service times are allowed with hard customer windows when the depot does not close and
  no hard duration or shift applies. Since the next stop is chosen after a service ends, a customer
  that can no longer be served in time is masked and ends unserved; for a priori routes this gives
  Errico et al.'s success probability (checked against their exact recursion, eq. 18-20).
- `time_windows.depot: ignore` (the depot has no window; instances cannot store an infinite closing
  time). Rejected with dynamic requests and planned routing; PyVRP re-optimization then leaves the
  depot open.
- `discrete_triangular: {low, mode, high, step}`: each grid point gets the probability of its cell
  under the continuous density (the paper does not state its discretization).
- Presets `tas2012-vrpsttw` (56 x 3) and `errico2018-vrptwst` (4 x 85); examples 16 and 17.

## Alternatives

- Per-traversal travel-time draws: exact for repeated arcs, but a scenario would need a sequence per
  arc; the papers' routes drive each arc at most once.
- Arriving at a customer and then finding its window closed (the a priori model's literal reading):
  needs the next stop committed before the service time is known. Skipping gives the same success
  probability for a priori routes and keeps FleetGym's decision timing.
- Lexicographic or rho-weighted objectives for Taş: rho uses C1 and C2 from the method's initial
  solution, so the preset uses the raw costs.

## Consequences

- New defaults (`early`, `depot`) changed the content hashes of the unreleased `-v1` families; the
  golden hashes were updated.
- Errico's "positive-skewed" medians (7.0, 63.0) are impossible for a triangle on the stated
  supports; the preset uses them as modes and says so.
