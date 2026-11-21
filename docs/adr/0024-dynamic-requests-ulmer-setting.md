# ADR 0024: Model features for the dynamic-request setting of Ulmer et al.

Date: 2026-11-20. Status: accepted (the user chose arrival epochs and mandatory early requests).

## Context

The DVRP paper example (M13) reproduces the VRPSSR of Ulmer, Goodson, Mattfeld and Hennig
(working paper, August 2016; Sections 2, 6.1 and the Appendix). `DVRP-v1` already covered the
arrival process (an `nhpp` profile gives lambda per T - 1 minutes on [1, T]), one vehicle, the
return by T and cheapest-insertion routing. Four things were missing:

1. clustered request locations (normal around two or three centers, weighted);
2. travel times rounded up to whole minutes (Euclidean kilometers at 25 km/h);
3. early-request customers that must be served (FleetGym let the agent reject them);
4. decisions at the vehicle's arrivals on the requests made since the last one (FleetGym decided
   each request when it was released).

## Decision

- `normal_mixture` distribution (`centers`, `std`, `weights`), any dimension; two-dimensional
  centers are request locations, one-dimensional ones a scalar.
- `network.time_rounding: none | ceil` on Euclidean networks; distances are unchanged. Rounding
  forgives a floating-point excess of 1e-9 relative. The validator applies it independently.
- `dynamics.initial_requests: mandatory`: the instance's customers are accepted at the start, in
  node order, by the routing rule (the paper sequences them by cheapest insertion too). The
  validator fails a rejection of such a request. With `next_node` it equals `instance`.
- `dynamics.initial_infeasible: error | serve_all` for mandatory requests that do not fit. `error`
  raises `InstanceError`. `serve_all` follows Ulmer, Mattfeld and Köster (Transportation Science
  2017, Section 3.1, the same problem): the early customers are planned without closing times,
  shift end or duration limit, the route may end late, and no other request can be accepted that
  day; an `initial_overtime` row (new event kind) records it. The validator then relaxes those
  limits for the day and fails any accepted later request; it does not re-check that the start
  was infeasible. The preset uses `serve_all` (the user's choice, 2026-11-20).
- `decisions.request_epochs: on_release | on_vehicle_arrival` for `accept_reject` and `assign`.
  Held requests are decided at the next arrival of any vehicle at a node, one by one in release
  order; while a vehicle waits with nothing planned they are decided at once; requests still held
  when no vehicle can arrive are decided before the episode ends.
- Preset `ulmer2016-vrpssr` (18 settings); example `15_dvrp_ulmer` writes the instances, each
  file one realization of the early customers with the late-request distributions.
- Fixed while doing this: literal parameters of a request-location distribution given by the
  instance (for example `uniform` with numbers) were looked up in the spec and not found.

## Alternatives

- Joint decisions on all requests of an epoch (the paper's `a` vector): an action space that grows
  exponentially; sequential decisions in release order, with the same information, can express
  every joint choice made by cheapest insertion except the dependence on later requests of the
  same batch. Kept sequential; stated in the preset's notes.
- Waiting as an explicit one-minute decision: the paper's heuristics disallow it; FleetGym's
  planned routing waits automatically when nothing is planned and decides arriving requests at
  once. Stated in the preset's notes.
- Rejected early requests with a large penalty instead of `mandatory`: the user preferred the
  paper's constraint.

## Consequences

- The new fields have defaults, so the content hashes of every registered `-v1` family changed
  (dynamics unchanged); the families are unreleased, so the golden hashes were updated without a
  version bump.
- The working paper assumes the early customers fit in one route; with a large area, low lambda and
  uniform locations only about 2% of its draws fit by T (FleetGym's cheapest insertion). Redrawing
  them was rejected because it biases that setting toward fewer early customers (about 68 instead
  of 75); with `serve_all` the instances keep the paper's Poisson draws.
- The two-cluster centers follow the working paper's text, (5, 5) and (5, 15); the 2017 paper
  describes its two clusters as symmetric to the central depot. Recorded in the preset's notes.
