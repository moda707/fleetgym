# EV shortest path with charging-station occupancy

`EVSPP-v1` is the problem of Dastpak, Errico, Jabali and Malucelli (arXiv 2305.11773): one electric
vehicle drives from an origin to a far destination and recharges on the way at public charging
stations, each shared with other EVs. It has its own environment, `EVShortestPathEnv`
(`fleetgym.make_ev("EVSPP-v1", instance=...)`; `fleetgym.make_any` and the presets pick it for
specs with `ev.shortest_path`), its own simulator and an independent validator. Design: proposal 0003, ADR 0026.

```yaml
id: EVSPP-v1
network: {type: euclidean, rounding: none, speed: 100.0}  # km and hours
ev:
  shortest_path:
    target_levels: 20                                      # action grid of target battery levels
    horizon: 72.0                                          # hours of exogenous activity per episode
    queues: {model: mm1k, initial: steady_state, full: leave}
decisions: {interface: ev_route, epochs: on_event, commitment: diversion}
objective: {drive: 1.0, wait: 1.0, charge: 1.0}           # the arrival time at the destination
```

## Instances

Schema `fleetgym/evspp/1`: node 0 is the origin, node 1 the destination, the others are stations
(`role`). Globals: `battery_capacity`, `consumption` (energy per km). Per station: `technology`,
`arrival_rate` (lambda), `service_rate` (mu), `queue_capacity` (kappa) and the charging function
`Phi` as three breakpoints (`charge_level_k`, `charge_time_k`: the time to charge from 0 to that
level), piecewise linear through `(0, 0)`. The EV leaves the origin fully charged. Instances from
which no chain of full-battery hops reaches the destination are rejected.

## Stations

Each station has one charger and an M/M/1/kappa queue of exogenous EVs: Poisson arrivals, exponential
charging, at most kappa EVs (exogenous EVs leave a full station). An episode's scenario holds, per
station, the EVs present at the start (from the stationary distribution) and every arrival of the
horizon with its charging time, each station on its own random stream. Queues are first come, first
served; the EV counts toward kappa once it has joined; with `full: leave` it cannot join a full
station (unless nothing else is reachable).

## Decisions and observations

Epochs: leaving the origin, any occupancy-indicator change while the EV drives or waits, arrival at
a station (the queue length is then seen), reaching the head of the queue, and the end of a charge.
The action space is `Discrete(max_nodes * (target_levels + 4))`: drive to a node (from wherever the
EV is: it may change its destination while driving), or charge at the current station up to a level
of the grid or one of its breakpoints (waiting first if the charger is busy). Observations give the
battery, position, status and queue seen, and per node its position, role, occupancy indicator and
the time since it last changed, and the queue parameters; exogenous arrivals stay hidden.
`env.fork(seed)` redraws every queue from its indicator (and the queue the EV sees), keeping the EV's
own place.

## Baselines

`EVGreedy` (the reachable station closest to the destination, charge to full) and `EVSteadyState`
(re-planning at arrivals, at the head of a queue and after charging, with the steady-state waiting
times `W_j = L_j / mu_j`, like the paper's benchmark). The preset `dastpak2023-evsppoi` has the
paper's 486 instances; example `18_evspp_dastpak` writes them.
