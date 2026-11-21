# Proposal 0002: stochastic travel and service times (M15)

Status: accepted by the user (2026-11-21); implemented in M15 (ADR 0025). One addition during
implementation: `time_windows: {depot: ignore}` states that the depot has no window (instances
cannot store an infinite closing time). Sources: Taş, Dellaert, van Woensel and de Kok, *Vehicle
Routing Problem with Stochastic Travel Times Including Soft Time Windows and Service Costs* (Beta
WP 364, 2012); Errico, Desaulniers, Gendreau, Rei and Rousseau, *The vehicle routing problem with
hard time windows and stochastic service times* (EURO J Transp Logist 7:223-251, 2018). All numbers
below are from these papers; anything they do not state is marked **assumed**.

## 1. The two settings

| | Taş et al. (2012) | Errico et al. (2018) |
| --- | --- | --- |
| Instances | Solomon, 100 customers, all six classes; capacity Q = 200 for every class (Section 5) | Solomon, demands and capacity ignored; first 25 customers of all 56 instances, first 50 customers of the 29 type-1 instances: 85 instances (Section 5.1) |
| Uncertain | travel time per unit distance T ~ Gamma(alpha, lambda), so T_ij ~ Gamma(alpha d_ij, lambda), E[T_ij] = d_ij (eq. 14-17) | service times: discrete triangular on a 0.1 grid, independent (Section 5.1) |
| Grid | (alpha, lambda) = (16, 0.0625), (1, 1), (0.0625, 16): CV 0.25, 1, 4 | Basic: R/RC {8.0..12.0} median 10, C {70.0..110.0} median 90, alpha = 0.95; Low-probability: alpha = 0.85; Large-support: R/RC {5.0..15.0}, C {45.0..135.0}; Positive-skewed: Large-support with "median" 7.0 (R/RC), 63.0 (C) |
| Time windows | soft, no waiting: service starts on arrival; early and late service are penalized (Section 3) | hard: wait if early, never start after b_i; the depot has no time window (Section 3.2) |
| Fleet | unlimited, homogeneous, fixed cost per vehicle | unlimited, homogeneous |
| Objective | c_d delay + c_e earliness + c_t distance + c_f vehicles + c_o overtime, with (c_d, c_e, c_t, c_f, c_o) = (1, 0.1, 1, 400, 5/6); overtime beyond w = 480 at the depot (Section 5) | distance, subject to P(all routes succeed) >= alpha; a route fails if it reaches a customer after its window (eq. 10) |
| Plans | a priori routes, all leaving at time 0 (the departure shift of Section 4.3 is part of their method) | a priori routes |

## 2. What FleetGym needs

### 2.1 Stochastic travel times (`uncertainty.travel_time`)

- The reserved source `travel_time` becomes available: `uncertainty: {travel_time: {revealed:
  on_arrival}}`, with the distribution from the instance (`distributions.travel_time`) or the spec.
- New distribution `gamma_per_distance: {shape: alpha, scale: lambda}`: the arc (i, j) takes
  Gamma(shape = alpha d_ij, scale = lambda), the sum of d_ij independent unit-distance times (Taş
  eq. 16-17). Other arc models (a multiplier on d_ij / speed) can be added later the same way.
- **Assumed:** one draw per arc per episode, pre-sampled into the scenario as a matrix
  (`travel_time`, its own random stream). An arc driven twice in one episode takes the same time.
  Taş's routes drive each arc at most once, so this matches them; it keeps scenarios small and makes
  common random numbers exact.
- Masks and observations use expected travel times; the realized time of an arc becomes known when
  the vehicle arrives. The validator replays the scenario's arc times.
- Scope: soft time windows, and no time limits other than soft ones. Hard windows, hard route
  duration or shift limits combined with stochastic travel times raise `NotInScopeError` (a vehicle
  could arrive after a closing time it committed to); no paper here needs it.

### 2.2 Early service under soft windows (Taş)

- `time_windows: {mode: soft, early: wait | serve}`. `wait` (the default) is today's behavior.
  `serve` starts service on arrival and charges `earliness = open - arrival` (new cost component).
  Lateness is charged as today.
- Overtime: `route_duration: {mode: soft}` with `max_route_duration = 480` in the instance gives
  `overtime` (exists).
- **Assumed:** the depot's window is not penalized (Taş state that it is the scheduling horizon and
  charge overtime against w = 480; Solomon's depot closes earlier than 480 in several classes). The
  example writes the depot's window as [0, infinity).
- **Assumed:** the objective weights are the paper's costs (c_d, c_e, c_t, c_f, c_o). Their rho
  weighting divides the two cost groups by C1 and C2, computed from their own initial solution, so
  it is a property of the method; the raw costs equal rho = 0.5 with C1 = C2.

### 2.3 Hard windows with stochastic service times (Errico)

- Today FleetGym rejects stochastic service times with hard time windows (`NotInScopeError`). This
  is lifted when the only hard time limits are customer windows: no depot closing time, no hard
  route duration or shift.
- What happens when a window cannot be met (the user's choice: count the failure, skip the customer,
  continue). In FleetGym a vehicle chooses its next stop after the current service has ended, so with
  deterministic travel it knows its arrival time: a customer it can no longer reach in time is masked
  and skipped, and ends `unserved`. For a priori routes (the `AprioriRoute` policy, one fixed order
  per vehicle) a route fails exactly when one of its customers is skipped, so the share of episodes
  without unserved customers estimates the paper's success probability (eq. 10). A vehicle never
  drives to a customer it would reach too late, so the distance of a failed route is slightly
  shorter than "drive there and find the window closed"; the success probability is the same.
- Evaluation reports the success rate (episodes with no unserved customer) next to the cost.
- New distribution `discrete_triangular: {low, mode, high, step}`. **Assumed:** each grid point
  gets the probability of its cell [x - step/2, x + step/2] under the continuous triangular density,
  so every listed support point has positive probability. The paper gives the support and the
  median but not the discretization.
- **Positive-skewed:** a triangular distribution on [5, 15] cannot have median 7.0 (the smallest
  possible median, with the mode at 5, is 15 - sqrt(50) = 7.93; on [45, 135] the smallest is 71.4,
  not 63.0). The only consistent reading is that 7.0 and 63.0 are the **modes**; the preset says so.
- **Assumed:** travel times are Solomon distances without rounding (the paper discretizes time in
  0.1 steps but does not say how travel times are rounded).
- alpha (0.95 or 0.85) is a constraint on plans, not part of the simulation; the preset records it
  as the reliability target that results are compared against.

## 3. Presets and examples

- `tas2012-vrpsttw`: 56 Solomon instances x CV {0.25, 1, 4}; example `16_vrptw_tas` writes the
  instances (Solomon download, capacity 200, depot window opened, `max_route_duration = 480`,
  `distributions.travel_time`).
- `errico2018-vrptwst`: families {basic, low_probability, large_support, positive_skewed} x sizes
  {25, 50} x the Solomon instances of Section 5.1; example `17_vrptw_errico` writes the instances
  (service-time distributions per class) and evaluates a priori routes, reporting success rates.

## 4. Tests

KS tests for `gamma_per_distance` (per-arc mean d_ij and variance alpha lambda^2 d_ij) and
`discrete_triangular`; stream independence (changing the travel-time distribution leaves demands and
service times unchanged); the validator recomputes arrivals from the scenario's arc times;
non-anticipativity with stochastic travel times; earliness and overtime against a hand-computed
route; an `AprioriRoute` success rate against Errico's exact recursion (eq. 18-20) on a small
instance, which gives an independent check of the simulator.
