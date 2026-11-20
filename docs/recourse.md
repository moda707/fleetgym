# Recourse: design for M5 and backlog

Status: implemented in M5 (2026-11-20); exact semantics in ADR 0013. User direction: ship standard
recourse rules (for example return to the depot and come back) and support user-written custom
rules; keep a list of later improvements and extensions (the backlog at the end of this file).

## What a recourse rule is
A **failure** happens when information revealed during execution makes the planned service
impossible. In v0.1 the only failure type is `capacity_failure`: the demand revealed at a customer
exceeds the vehicle's remaining load. The spec says which rule handles each failure type:

```yaml
recourse:
  capacity_failure: return_to_depot                 # short form
  # or: {rule: return_to_depot, partial_service: true, on_exact_depletion: continue}
```

A rule receives the failure and the **information state** (revealed information only, never the
scenario), and returns **primitive decisions** that the simulator executes like any other decision:
partial service, a trip to the depot to restock, a revisit, outsourcing. Because the simulator
executes them, travel, time, load and cost accounting are the same as for agent decisions, and the
event log records everything. Extra costs (for example an outsourcing fee) go to named cost
components in `info["costs"]`.

```python
class RecourseRule(ABC):
    def serve_first(self, failure: CapacityFailure, info: InformationState) -> bool:
        """Serve what fits before resolving (default: the `partial_service` param)."""

    @abstractmethod
    def on_failure(self, failure, info) -> Literal["return_to_depot", "outsource"] | None:
        """How to resolve the rest; None hands the choice to the policy (recourse epoch)."""

    def after_service(self, vehicle: int, info: InformationState) -> bool:
        """Restock preventively after completing a customer (default: never)."""
```

`CapacityFailure` carries the vehicle, the customer, the time, the demand still to deliver, the
free capacity and the capacity. Two rules hold for every rule: an empty vehicle always serves what
fits (so demand above capacity cannot loop), and each visit that delivers something takes the
customer's service time.

## Standard rules in M5
| Rule | Behavior |
| --- | --- |
| `return_to_depot` | Classical detour-to-depot recourse: restock at the depot, come back, continue the planned route. |
| `outsource` | The remaining demand is served by a third party at a cost (`outsourcing` component); the vehicle continues. |
| `agent_decides` | The simulator opens a **recourse epoch** and the policy chooses among the allowed options (`options: [return_to_depot, outsource]`), with a mask. This is the dynamic setting. |

The conventions that change expected costs (plan Q20 a to c) are **explicit parameters**, so no single
convention is built in silently:

| Parameter of `return_to_depot` | Values | Default |
| --- | --- | --- |
| `partial_service` (Q20c) | `true`: serve what fits, go to the depot, come back and finish. `false`: go to the depot first, come back, serve everything. | `true` |
| `on_exact_depletion` (Q20a) | `continue`: an emptied vehicle continues and fails at the next customer with positive demand. `restock`: return to the depot right after a customer that empties the vehicle. | `continue` |
| demand above capacity (Q20b) | The vehicle makes as many depot round trips as needed (each serves up to the capacity). | always |

Configs that reproduce a paper set these parameters explicitly and cite the paper's convention.

## Custom rules
A custom rule is a `RecourseRule` subclass registered under a name, from user code or a package
entry point (group `fleetgym.plugins`), and then referenced from YAML like a standard rule:

```python
@fg.register_recourse("my_lab.near_depot", params=MyParams)
class NearDepot(fg.RecourseRule):
    def on_failure(self, failure, info):
        near = info.distances[failure.node, 0] <= self.params.radius
        return "return_to_depot" if near else "outsource"
```

- The spec hash covers the rule's name and parameters, not its code. Run metadata records the
  plugin's package and version.
- Rules see the information state only, so they cannot use future information; the
  non-anticipativity checker covers them like any policy.

## How correctness is tested
- **Validator (independent):** it checks the physics of whatever the rule did (loads never negative,
  every unit of revealed demand served once or outsourced, depot restocks, travel and cost accounting)
  from the event log. For the standard rules it also re-implements the rule and checks that it was
  followed. For custom rules it checks the physics only; a rule may ship its own checker.
- **Exact expectation:** `fleetgym.testing.exact.expected_cost(spec, instance, run_episode)`
  enumerates every scenario of finite-support distributions, plays a policy (for example
  `fleetgym.baselines.apriori.AprioriRoute`) on each, and returns the probability-weighted cost. M5
  compares it with an independent dynamic program to 1e-9 for `return_to_depot` (all parameter
  combinations) and `outsource`. Authors of custom rules use the same enumerator against their own
  closed form.
- Monte Carlo sanity check for Poisson demands; determinism; scenario replay.

## Backlog (later improvements and extensions)
1. **Preventive restocking** (Yang et al. 2000): threshold rules and the optimal threshold policy as an
   `after_service` rule (`VRPSD-Restock-v0`).
2. **Split as a new request**: leave the unserved remainder as a new dynamic request (needs M6).
3. **Time-window failures**: serve late at a penalty, or skip (failure type `time_window_failure`).
4. **Battery shortfall**: emergency charging or towing (`energy_failure`; EV module, out of scope).
5. **Outsourcing variants**: carrier capacity, per-unit vs per-customer fees, partial outsourcing
   (to be aligned with the VRP-SDO model, deferred after v0.1).
6. **Duration limits with recourse** (Goodson et al. 2013), where a detour can break the route duration.
7. **Multiple depots or replenishment points**: restock at the nearest facility.
8. **Stochastic customers** (VRPSC): skip absent customers.
9. **Validator hooks** so custom rules can register an independent checker (today the validator
   checks the physics of custom rules, and the behavior of built-in rules only).
10. **Code identity for custom rules** in the spec hash (for example a declared rule version).
11. **Recourse under communication delay**: the decision reaches the vehicle after a lag.
12. **Forks during a service in progress**: a fork keeps the scheduled end of a service whose time
    is revealed on completion; resampling it conditional on the elapsed time would be exact.
13. **Stochastic service times with hard time constraints** (time windows, duration, shift).
