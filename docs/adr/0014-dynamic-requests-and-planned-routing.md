# 0014: Dynamic requests and planned routing

Status: accepted (2026-11-20). Adopts plan Q24 to Q27 (recommendations). Two choices filled gaps in
the design and are flagged for the user: the cancellation timing and `initial_requests`.

## Requests of a day
- **Node slots**: depots, then requests known at the start (the instance's customers with
  `dynamics.initial_requests: instance`, the default), then `max_requests` slots for requests that
  arrive during the day, in arrival order. `episode_instance(base, scenario)` builds this instance;
  observations are padded to it, and unreleased slots are hidden (`node_valid = 0`, all features 0).
  With `initial_requests: none`, the instance's customers only serve as data (for example for the
  location KDE).
- **Arrivals**: `poisson` (exponential gaps) or `nhpp` (piecewise-constant rate, thinning) on
  `[shift start, end)`, where `end` is the earliest of the cutoff, the end of the rate profile, the
  depot's closing time and the shift end. Requests after the cutoff are not generated (Q25).
- **Attributes**, each from its own named stream: location from `kde` (two-dimensional data) or
  `uniform` (a square); demand and service time from scalar distributions (0 when absent); the time
  window (with the `time_windows` constraint) opens at release and closes `time_window_width` later,
  capped at the depot's closing time.
- **Cancellations** (flagged for the user): `cancellations` is a 0/1 distribution per request; a
  cancelling request cancels at a time uniform between its release and its deadline (window close,
  or the end of the arrival window). The design names the option but not the timing. A cancellation
  after the service has started has no effect. A vehicle already driving to a cancelled request
  arrives, serves nothing and continues. The user (2026-11-20) has no reference model either: the
  timing stays as is and is listed as future work (a cancellation hazard or patience model).
- **Overflow** (Q24): `max_requests` is required; `overflow: error` (default) raises
  `RequestOverflowError`; `overflow: reject` rejects extra requests on arrival (`reject` rows with
  `component: overflow`, one `rejected` cost each). Their release times are the scenario's
  `request.overflow` source (the only source of variable length).
- **Dynamism metrics**: `degree_of_dynamism` (share released after the start) and
  `effective_degree_of_dynamism` (mean relative release time). The definitions are written in the
  module docstring; check them against Lund et al. (1996) and Larsen et al. (2002) before comparing
  with published values.
- Uncertain parameters on dynamic requests (`uncertainty` together with `dynamics`) and dynamic
  requests with `next_node` (which needs wait and reposition decisions) are not in this version.

## Planned routing (`accept_reject`, `assign`)
- A request epoch per released request (requests known at the start are decided at the start, in
  node order, before any vehicle moves). `accept_reject` is `Discrete(3)`: reject, accept, outsource
  (masked without an `outsourcing` constraint; Q26). `assign` is `Discrete(V + 1)`: a vehicle, or
  reject as the last action (Q26).
- `routing: cheapest_insertion` lives in `sim/routing_rules.py` (Q27): the position adding the least
  distance among those that keep the whole remaining plan feasible (capacity, hard windows with
  waiting, route duration, shift end, depot close); ties go to the lowest vehicle, then the earliest
  position. Accept is masked exactly when no feasible insertion exists; the test checks this against
  the independent Gate 1 route oracle.
- **Vehicles**: drive first (an idle vehicle with a plan leaves for its next stop at once; the stop
  is then committed). With an empty plan a vehicle waits where it is (Q26), but leaves for the depot
  at the latest time that still respects the depot's closing time, the shift end and the route
  duration (a wake-up timer); vehicles driving home are not offered new requests. When no request can
  arrive any more, vehicles return to the depot (or retire if they never left).
- The `outsourcing` constraint is an option here: outsourcing a request costs its `value`
  (per request; instance-referenced costs cover only the instance's own customers).
- A fixed or limited fleet and `on_event` epochs are required; the objective needs a `rejected`
  weight (or outsourcing). `waiting: forbidden` is not supported.
- **Forks**: `fork(seed)` keeps every request released so far and redraws the future: new arrivals
  after now, and the cancellation of released requests conditioned on not having cancelled yet.

## Validation
The validator rebuilds the episode instance from the base instance and the scenario. It checks
release times, one decision per released request, overflow rejections, cancellation times, that a
vehicle only drives to requests accepted for it and only serves accepted, uncancelled requests,
that every accepted request is served (or cancelled), and the costs.

## Consequences
- Observation node features grow to 10 (an under-decision flag). `Epoch.vehicle` is `None` at
  request epochs. Interfaces receive `max_vehicles`; invalid-action messages come from the
  interface (`ActionInterface.explain`).
- `DVRP-StochReq-v0` is `experimental` (fixed fleet of 4, accept_reject); its parameter values are
  illustrative.
