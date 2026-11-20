# 0016: Multi-agent environments

Status: accepted (2026-11-20). Adopts plan Q28 (recommendation).

## Decision
- One agent per vehicle (`vehicle_<i>`), on the same simulator as the single-agent env. Requires
  `next_node`, `on_event` epochs and a fixed or limited fleet whose size is known when the env is
  built (a number, or an `instance=`).
- **AEC** (`FleetGymAECEnv`): the agent whose vehicle has a decision epoch acts; a vehicle that is
  done (route ended or retired) is terminated and removed through PettingZoo's dead step; other
  agents observe with an all-zero mask.
- **Parallel** (`FleetGymParallelEnv`): every vehicle that must decide at the current time acts at
  once (`Simulator.ready_epochs`, `Simulator.apply_many`). Busy vehicles send the **no-op**, an extra
  last action, which is their only valid action; a deciding vehicle cannot send it. Decisions are
  executed in vehicle order against the state left by the previous ones.
- **Conflicts** (two vehicles choosing the same customer): `priority` (lowest index), `distance`
  (nearest, ties by index), `lottery` (the `agents.conflict` stream of the episode seed, keyed by
  epoch and customer, so it is reproducible and needs no state), or `reject` (nothing is applied;
  every agent receives `-conflict_penalty`, logged as an `invalid_action` cost). A losing vehicle
  keeps its decision and is asked again at the same time. Conflicts are logged (`conflict` rows).
- **Rewards**: `global` (every live agent receives the total), `individual` (each vehicle's own
  costs plus an equal share of costs attributed to no vehicle, so individual rewards sum to the
  total), `mixed` (`reward_mix` blend). The simulator attributes each cost to a vehicle where it can
  (`Transition.vehicle_costs`, `info["vehicle_costs"]`).
- `state()` / `state_space`: all nodes, all vehicles and global features, flattened, for a
  centralized critic.
- Configs: explicit files only (Q28): `MA-CVRP-v0` and `MA-DVRP-v0` (dynamic requests with wait,
  ADR 0015), both Parallel by default. They are not registered with Gymnasium; `fleetgym.make`
  refuses them and points to `fleetgym.make_multi(id, api=...)`. `REGISTRY.ids(agents=...)` filters
  single- and multi-agent configs.
- Reset options other than `instance` and `scenario` are ignored by the multi-agent envs, because
  PettingZoo's API test resets with arbitrary options.

## Verification
PettingZoo `api_test` and `parallel_api_test` on both configs; an AEC env with one vehicle equals
the single-agent env; AEC and Parallel give the same total cost on the same scenario when vehicles
choose disjoint customers; tests for each conflict rule and reward mode; validated dynamic episodes.

## Consequences
- Heterogeneous fleets, communication channels and agents joining or leaving are after v0.1.
- Training wrappers (parameter sharing for SB3, BenchMARL adapters) come with M9.
