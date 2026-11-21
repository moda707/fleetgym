# Simulation and decisions

FleetGym simulates with a **discrete-event engine**. Time jumps from event to event (a vehicle
arrives, a service starts or ends, a request arrives), and the agent is asked only at **decision
epochs**.

## Epochs and interfaces

| Interface | The agent chooses | Action space |
| --- | --- | --- |
| `next_node` | the next node of the vehicle that must decide | `Discrete(max_nodes)` (`+1` wait with dynamic requests) |
| `accept_reject` | reject, accept or outsource the request that just arrived | `Discrete(3)` |
| `assign` | the vehicle for the request that just arrived, or reject | `Discrete(V + 1)` |

With `epochs: sequential` one vehicle builds its whole route before the next starts (the classical
construction setting of neural combinatorial optimization). With `epochs: on_event` all vehicles move
concurrently and decide when they become free.

## What the agent sees

Constraints, masks and observations read the simulator only through an **information state**: what
is known now. Unrevealed values appear as forecasts, and the observation says which values are known.
A mask allows exactly the actions that keep the plan feasible given what is known, including the
return to the depot; under uncertainty an allowed action can still fail later, and recourse handles
that.

## Rewards

The reward of a step is minus the weighted cost realized since the previous epoch; the components
are in `info["costs"]`. Summed over an episode, the rewards equal minus the total cost recomputed by
the validator (tested). Lexicographic objectives are scalarized with a factor of 10⁶ per level.

Objectives that maximize something use a revenue component with a negative weight:
`{served_demand: -1}` rewards each unit of demand when it is delivered (`VRP-VCSD-v0`). Only revenue
components may have negative weights.

## Snapshots

`env.get_state()` and `env.set_state()` copy everything, including random generator states, and
`env.clone()` makes an independent copy that shares the true future. Planners must use
`env.fork(seed)` instead, which keeps the revealed past and redraws the hidden future
({doc}`uncertainty`).
