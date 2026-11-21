# ADR 0026: The EV shortest path with charging-station occupancy (M14)

Date: 2026-11-21. Status: accepted (the user approved proposal 0003 and answered its open points).

## Context

The user brought the EVSPP-OI of Dastpak, Errico, Jabali and Malucelli (arXiv 2305.11773) into
scope. It is not a vehicle routing problem: one EV, no customers, a battery, charging functions and
queues of exogenous EVs whose binary occupancy indicators are observed in real time. The `ev` spec
section and the `charge` interface were reserved. Design: `docs/proposals/0003-...md`.

## Decision

- A typed `ev.shortest_path` section (`target_levels`, `horizon`, `queues: {model, initial, full}`)
  inside the otherwise reserved `ev` section; family `EVSPP-v1`; interface `ev_route`; cost
  components `drive`, `wait`, `charge`; `decisions.commitment: diversion` is allowed for it.
- Its own simulator (`fleetgym.sim.ev`), environment (`fleetgym.envs.ev_env.EVShortestPathEnv`,
  made by `fleetgym.make_ev`; `make_any`, the presets and the Gymnasium registration choose it for
  EV specs, while `fleetgym.make` stays typed for routing and points EV specs to `make_ev`, as it
  points multi-agent specs to `make_multi`) and independent validator (`fleetgym.testing.ev_validator`, called by
  `validate`; it obeys the validator's layer rule, which now covers both modules). The environment
  exposes the surface of `FleetGymEnv` the tools use (masks, log, costs, fork, clone, state), so the
  generic tests run on it.
- Instance schema `fleetgym/evspp/1`: node 0 origin, node 1 destination, stations with technology,
  lambda, mu, kappa and three charging breakpoints (the paper's functions all have three). The JSON
  reader now keeps string node fields of a schema.
- Scenarios hold, per station, the exogenous EVs present at the start and the arrivals of the horizon
  with their charging times, each station on its own stream (`ev.station.<node>`).
- The user's choices where the paper is silent: the EV leaves fully charged; queues start in their
  steady state; the EV cannot join a full station (`full: leave`), except when nothing else is
  reachable, so an episode can always go on.
- Actions: a node to drive to, or a target level at the current station from a grid of B / 20 plus
  the station's breakpoints (the paper's target is continuous).
- Forks redraw each queue from its indicator (stationary distribution given busy) and the queue the
  EV sees, with fresh charging times and future arrivals; a non-anticipative approximation of the
  posterior (it ignores how long an indicator kept its value).
- Instances from which the destination cannot be reached with full-battery hops are rejected.
- Baselines `EVGreedy` and `EVSteadyState` (the paper's steady-state benchmark in a simple form,
  charging to full); preset `dastpak2023-evsppoi` (486 points); example `18_evspp_dastpak` with the
  nested samples of Section 5.3.

## Alternatives

- Extending `FleetGymEnv`: its observations, interfaces and fleet logic are built around vehicles
  and customers; a separate environment keeps both simple.
- A continuous target level (`Box` action): harder for discrete RL and masks; the grid includes the
  breakpoints, where optimal charging amounts of piecewise-linear functions lie.
- An exact posterior for forks (a hidden Markov model over queue lengths given the indicator
  history): possible later; the approximation is documented.

## Consequences

- New spec defaults and the new family changed the golden hashes of the unreleased `-v1` families.
- `run_episode`, the evaluation harness and the policy protocol accept both environments.
- Table 3 of the paper lists PM1 as 55% Slow while its text says PM2; Section 5.3 (PM2 = 45, 45,
  10) settles it, recorded in the preset.
