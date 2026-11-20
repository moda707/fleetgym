# 0009: Simulator semantics, rewards and environment conventions (M3)

Status: accepted (2026-11-20). Adopts plan recommendations Q12 to Q17 while the user was
unavailable; each can be revisited.

## Timing and costs
- Vehicles start idle at the depot at the shift start (`fleet.shift[0]`, else 0) and depart as soon as
  they are dispatched (no delayed departure in v0.1).
- `distance` and `time` (travel time) are charged at departure; `vehicles` (1) at a vehicle's first
  departure. A dispatched request becomes `ASSIGNED`, so no other vehicle can be sent to it.
- Service starts at `max(arrival, open)` when time windows allow waiting (the gap is charged as
  `waiting`), else at arrival. Requests are `SERVED` at service start; the vehicle is idle at
  service end. Soft windows charge `lateness = max(0, start - close)` at service start.
- Arriving at the depot ends the route (single trip). Route duration is return time minus departure
  time and includes waiting and service.
- At the end, each pending request becomes `UNSERVED` and costs 1 `unserved`.
- Events at the same time run in a fixed order (arrival, service start, service end, then insertion
  order), and all events up to time `t` run before an epoch at `t`.

## Epochs and fleets (Q12)
- `sequential`: one vehicle routes until it returns; the next starts at the shift start. Valid only
  for static problems, where vehicles do not interact; the clock restarts per vehicle, so time is
  monotone per vehicle, not globally.
- `on_event`: vehicles act concurrently in time order (ties by index).
- An unlimited fleet requires `sequential` epochs and exactly one vehicle type; new vehicles are
  created as earlier ones finish.
- A vehicle at the depot with an empty route decides only if some pending customer is feasible for
  it; otherwise it retires automatically. For an unlimited fleet this stops vehicle creation (they are
  identical), which prevents livelocks.
- Retiring at the depot (choosing the depot before leaving) is allowed for fixed and limited fleets,
  and for unlimited fleets only when nothing else is feasible.
- **Q13**: a fixed or limited fleet without outsourcing must weight `unserved` in the objective;
  otherwise leaving requests unserved would be free.

## Masks and constraints (Q16)
- Constraints see only the `InformationState` (what is known now), never the scenario.
  `Constraint.init_state(info)` and `penalty(state, info, event)` replace the design's
  `init_state(spec, scenario)` and `penalty(state, event)`.
- Masks only cover customers; returning to the depot is never masked. Hard constraints therefore look
  ahead to the depot: hard time windows check the depot's closing time, route duration checks the
  return, and `fleet.shift` acts as a hard return deadline.
- Fleet size is structural (vehicles exist or not), not a mask.
- An episode always has at least one step. If nothing can be decided at reset, action 0 is the only
  valid action and ends the episode; costs realized before the first decision are added to the first
  step's reward, so the return always equals minus the weighted total cost.

## Rewards (Q15)
- Reward = minus the weighted sum of costs realized during the step. A lexicographic objective
  `[c_1, ..., c_L]` uses weights `1e6^(L-k)`; all components are reported in `info["costs"]` and
  `info["total_costs"]`, and benchmarks compare them lexicographically.

## Environment
- `invalid_action`: `mask` (default) raises `InfeasibleActionError` with the violated constraint;
  `repair` maps to the feasible node nearest (by distance) to the chosen one, ties to the lowest
  index; `penalize` leaves the state unchanged and charges `invalid_action_penalty`. Gymnasium's
  `check_env` steps random unmasked actions, so it is run with `repair`.
- Seeds: `reset(seed=s)` makes `s` the episode seed; later unseeded resets use seeds derived from
  `(s, episode index)`. The instance is drawn from the episode's `instance` stream (ADR 0007,
  `StreamFactory`: streams keyed by a SHA-256 of the source name).
- Observation and vehicle slots are fixed per environment: `max_nodes` is inferred from the instance
  source (or given); `max_vehicles` defaults to the larger of `max_nodes` and the fleet size known
  at construction. Exceeding them raises `ConfigError`.
- Box bounds are the finite float32 range (Gymnasium warns about infinite bounds).
- `get_state`/`set_state` are exact (simulator, constraint states, event log, seeds, RNG state);
  `clone()` shares the future scenario, so planning methods must use `fork` (M5).
- Gymnasium 1.x no longer loads plugin entry points, so envs register when `fleetgym` is imported;
  `gymnasium.make("fleetgym:fleetgym/CVRP-v0")` imports it on demand.

## Event log (Q17)
Built in M3 (not M10): one row per event with typed nullable columns, Parquet metadata with spec hash,
instance hash and episode seed. It is the validator's only input besides spec and instance.
