# 0021: VRP-VCSD: split recourse, duration limits and served-demand rewards

Status: accepted (2026-11-20).

## Context
The user asked for an example per environment that reproduces the problem setting of a published
paper, starting with their own. The VRP with a highly variable customer basis and stochastic demands
(Dastpak, Errico and Jabali, arXiv 2109.10200) needs things v0.1 did not have: a duration limit
together with recourse (backlog item 6), a remainder left open for any vehicle instead of a forced
detour, restocking whenever the vehicle chooses, an objective that maximizes served demand, and a new
customer set each day. The user chose to implement them (rather than approximate the setting) and to
also register the paper's VRPSD setting (Section 6.2, after Goodson, Thomas and Ohlmann 2016). All
numbers come from the paper's text (Section 3.1, 6.1.1, 6.2.1, Appendices B.1.1 and B.1.2).

## Decision
- **Recourse rule `split`** (the name was reserved): at a capacity failure the vehicle serves what
  fits and the customer is pending again with the demand left (`failure` row, component `split`).
  A vehicle with no free capacity may only go to the depot (`RecourseRule.must_restock`). Every trip
  to the depot during the day restocks and the route goes on (`RecourseRule.restock_trips`, as with
  dynamic requests in ADR 0015). `end_early: false` (the paper's condition) lets a vehicle at the
  depot end its day only when no customer is reachable (`RecourseRule.may_end`).
- **Duration limits with recourse**: allowed for rules that never force travel
  (`RecourseRule.no_forced_travel`; `split`). A vehicle at a feasible customer can always return in
  time, so no recourse can break the limit. `return_to_depot` and `agent_decides` still raise
  `NotInScopeError` with duration, shift or time-window limits.
- **Open demand**: `InformationState.open_demands` is the demand left at pending customers whose
  demand is revealed (the delivered part is known). The capacity mask and observation column 2 use
  it. In earlier environments a pending customer is never partly served, so nothing changes there.
- **Revenue component `served_demand`**: each service records the quantity delivered. It is the one
  component that may have a negative weight (`REVENUE_COMPONENTS`): `{served_demand: -1}` rewards
  delivered demand when it is delivered, as in the paper's reward. It cannot be a lexicographic level.
  A fixed fleet may use it instead of an `unserved` weight. Rule 9 (sum of rewards equals minus the
  validator's cost) still holds.
- **Distributions** `discrete_uniform` (integers `center +- w`, `w = min(half_width, center - floor)`,
  0 where `center <= floor`) and `scaled_categorical` (`scale * factors[i]` with `probs[i]`).
- **Generator `zones`**: a grid over the square; per active zone a count of customers, uniform
  locations in the zone, demands from a discrete distribution. An instance needs a customer, so the
  counts are redrawn when all zones are empty (probability 0.5^15 at the paper's Very Low density).
- **Instance source `collection`**: one instance of a shipped manifest (downloaded and verified),
  optionally its first `n` customers and without time windows or service times.
- **Configs** `VRP-VCSD-v0` (Moderate density, Q = 50) and `VRPSD-Goodson-v0` (Medium limit, Q = 50,
  Moderate variability); the other instances of the paper are overrides, listed in each YAML.

## Differences from the paper, and assumptions
- The paper's Figure 4 zone layout is not in its text: the 15 active zones are a fixed random draw
  (`numpy.random.default_rng(2109).choice(25, 15, replace=False)`, sorted), chosen by the user.
- Vehicles deciding at the same time decide in index order; the paper picks one at random.
- `VRPSD-Goodson-v0`: Euclidean distances without rounding, no service times and no time windows
  (the paper's model has travel times only; not stated explicitly).

## Alternatives
- Approximating the setting with v0.1 features (forced return-to-depot recourse, no duration limit,
  an unserved-customer penalty): rejected by the user.
- `unserved_demand` charged at the end instead of a revenue: same optimum, but a delayed reward
  unlike the paper's; and it needs no negative weights. Rejected for fidelity.
- A new request for the remainder (the reserved description of `split`): needs dynamic requests and
  changes request identities; keeping the same request open is simpler and equivalent.

## Consequences
- The validator re-implements the rule: a failure leaves the vehicle full, a full vehicle goes to the
  depot, depot trips restock, a partly served customer may end unserved, and the revenue equals the
  quantities delivered. It does not check `end_early` (a policy restriction, enforced by the mask).
- Tests that build every registered config give `VRPSD-Goodson-v0` the `network` marker (it
  downloads R101); offline tests run it on a local instance.
