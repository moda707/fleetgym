# Recourse

A **capacity failure** happens when a revealed demand does not fit in the vehicle. The spec says how
to handle it:

```yaml
recourse:
  capacity_failure: {rule: return_to_depot, partial_service: true, on_exact_depletion: continue}
```

| Rule | Behavior |
| --- | --- |
| `return_to_depot` | serve what fits, restock at the depot, come back (the classical detour) |
| `outsource` | the rest goes to a third party at `per_customer + per_unit * quantity` |
| `agent_decides` | the policy chooses at a recourse epoch (depot = restock, the customer = outsource) |

The conventions that change expected costs are parameters: `partial_service` (serve what fits before
restocking, or restock first) and `on_exact_depletion` (an exactly emptied vehicle continues, or
restocks right away). An empty vehicle always serves what fits, so demand above capacity cannot loop.
Each visit that delivers something takes the customer's service time.

Expected costs are verified exactly: over all 1,024 scenarios of a small instance, the environment's
expected cost of an a priori route equals an independent dynamic program to 1e-9 for every
combination of these parameters.

Write your own rule by subclassing `fleetgym.RecourseRule` (see
{doc}`../guides/build-your-own-variant`). The design and the list of planned extensions are in
[docs/recourse.md](https://github.com/moda707/fleetgym/blob/main/docs/recourse.md).
