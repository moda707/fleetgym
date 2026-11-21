# Proposal 0003: the EV shortest path with charging-station occupancy (M14)

Status: accepted by the user (2026-11-21), with the answers recorded in Section 3. Source: Dastpak, Errico, Jabali and Malucelli, *Dynamic
Routing for the Electric Vehicle Shortest Path Problem with Charging Station Occupancy
Information* (arXiv 2305.11773v1, 2023), Sections 3 and 5.1. Numbers are from the paper; anything
it does not state is marked **open** (a question for the user) or **assumed**.

## 1. The setting (paper)

- One EV drives from origin o to destination d over public charging stations (CSs); arcs are
  fastest paths with deterministic travel time t_ij and energy e_ij; battery capacity B; partial
  charging; the charging time from level 0 to sigma is a concave piecewise-linear Phi_i (Montoya
  et al. 2017), so charging from s1 to s2 takes Phi_i(s2) - Phi_i(s1) (Section 3.1).
- Each CS has one charger of technology f_i (Slow, Normal, Fast) and is an M/M/1/kappa queue of
  exogenous EVs: arrival rate lambda_i, service rate mu_i (by technology), capacity kappa;
  exogenous EVs do not join a full queue.
- Occupancy indicator q_i(t) in {0, 1}: 0 if nobody is charging or waiting. The EV receives every
  change of every indicator in real time; the queue length is seen only on arrival at the CS.
- Decision epochs: leaving o, any indicator change, arrival at a CS, the CS becoming free after
  waiting, the target level reached. No decisions while charging. Actions (Eq. 1): (d) when
  e_cd <= b, else (x, y): a CS x with e_cx <= b and a target level y in [b - e_cx, B] when leaving
  x; staying at the current CS is waiting. The EV may change its next node while driving (the
  paper's "deviations while driving"), from its current position c_k.
- Objective: the arrival time at d (driving + waiting + charging), Eq. 2.
- Instances (Section 5.1): a 350 x 700 km region, o = (0, 0), d = (350, 700); Euclidean distances,
  100 km/h; EVs of Table 1 (Peugeot I-on 16 kWh, 0.125 kWh/km; BMW i3s 42.2 kWh, 0.165; Renault Zoe
  52 kWh, 0.165) with the charging breakpoints of Table 2 (85, 95, 100% state of charge);
  densities of 2, 6, 10 CSs per 100 x 100 km (49, 147, 245 CSs), locations uniform; technology
  mixes PU1-PU3, PM1-PM3 (Table 3); mu = 0.28, 0.56, 1.12 per hour; utilization 40, 65, 90%,
  lambda = rho mu (Table 4); kappa in {1, 2, 3}; 486 instances; the experiments use the 18 of
  Table 6 with 15 daily scenarios each. Locations are nested across densities, and technology
  assignments across PM1-PM3 (Section 5.3).

## 2. FleetGym design

### 2.1 Model family `EVSPP-v1`

A new family on the reserved `ev` section and a new simulator (`EVShortestPathSimulator`, L4),
next to the next-node and planned-routing simulators:

```yaml
id: EVSPP-v1
ev:
  battery: from_instance            # B, the initial level and the charging functions
  stations: {queue: mm1k, occupancy: indicator}   # exogenous EVs; binary indicators in real time
  target_levels: 20                 # action grid of target levels (see 2.3)
decisions: {interface: ev_route, epochs: on_event, commitment: diversion}
objective: {duration: 1.0}          # new cost components: drive, wait, charge (duration = sum)
```

### 2.2 Instances (schema `fleetgym/ev/1`, extends the core)

Nodes: `kind` (origin, destination, station), `x`, `y`, per station `technology`, `lambda`, `mu`,
`kappa`. Globals: `battery_capacity`, `initial_battery`, `consumption` (kWh per distance unit),
`speed` comes from the network. A table of charging functions per technology (breakpoints of
level and time). Travel time and energy are derived from the Euclidean distance
(`t = d / speed`, `e = consumption * d`), as in Section 5.1.

### 2.3 Actions and observations (`ev_route`)

- `Discrete(n_nodes x (target_levels + 1))`: a next node and a target level from a grid of
  `target_levels` equal steps of B plus the charging breakpoints, or "leave as is". Masks allow
  only reachable nodes (e_cx <= b) and levels >= arrival level; choosing the current CS while
  queued is waiting. **Assumed:** a grid of levels instead of a continuous y (RL-friendly; the
  breakpoints are included so the paper's optimal charging amounts are reachable).
- Observation: battery, position, status (driving, waiting, charging), time, and per station
  (q_i, g_i = time since q_i last changed, x, y, technology); the queue length of the current CS
  once arrived. The exogenous arrivals and services are hidden.

### 2.4 Exogenous queues and scenarios

A scenario holds, per station, the exogenous arrival times and service durations of the day
(variable-length sources), sampled from each station's own stream (keyed by the station). The
queue is replayed from them exactly, with the EV inserted FIFO when it joins; **assumed:** the EV
counts toward kappa for exogenous EVs once it has joined. Forks redraw the unrevealed future of
every queue given what is known (the indicator history, and the queue seen at the current CS).

### 2.5 Validation and baselines

The validator replays the queues from the scenario, the battery (never below 0, charging only up
to B), charging times from Phi, waiting from the queue, and checks the arrival time at d equals the
total duration. Baselines: a steady-state re-optimizer like the paper's RSS (expected travel plus
steady-state waiting W_j = L_j / mu_j, charging decisions by label setting over the breakpoints;
implemented in FleetGym, no new dependency) and a greedy "nearest reachable station, charge to
full". Preset `dastpak2023-evsppoi` with the 486-point grid; example `18_evspp_dastpak` writes the
instances with nested locations and technologies.

## 3. Decisions on the open points (the user, 2026-11-21)

1. The EV leaves o with a full battery (B).
2. Each queue starts the trip in its steady state: the initial number of exogenous EVs is drawn
   from the M/M/1/kappa stationary distribution (with the remaining service and inter-arrival
   times exponential, by memorylessness).
3. The EV cannot join a full queue (kappa EVs present): waiting there is masked, it must leave.
