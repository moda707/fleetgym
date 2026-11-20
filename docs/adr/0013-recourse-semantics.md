# 0013: Recourse semantics

Status: accepted (2026-11-20). Implements the design agreed with the user (`docs/recourse.md`,
plan Q20): standard rules plus user-written rules, with the conventions that change expected costs
as explicit parameters.

## Failure and resolution
- A **capacity failure** is detected at service start: the customer's remaining (true) demand
  exceeds the vehicle's free capacity.
- The rule decides whether to **serve what fits first** (`serve_first`, from `partial_service`).
  **An empty vehicle always serves what fits**, whatever the rule says; otherwise a demand above
  capacity with `partial_service: false` would restock forever. A rule that would still restock an
  empty vehicle raises `RecourseLoopError`.
- After that service (or at once, if nothing was served), the rule resolves the rest:
  `return_to_depot` (drive to the depot, restock to load 0, drive back, check again) or `outsource`
  (the rest goes to a third party at `per_customer + per_unit * quantity`, component
  `outsourcing`). Returning `None` opens a **recourse epoch** for the policy (`agent_decides`).
- After a customer is complete, `after_service` may send the vehicle to restock preventively. The
  built-in use is `on_exact_depletion: restock`: when the vehicle is exactly full and customers are
  still pending. The vehicle then waits at the depot for a decision; choosing the depot there ends
  the route.

## Conventions
- **Service time** applies to every visit that delivers a positive quantity; a visit that delivers
  nothing takes no service time.
- **Restock trips** do not end the route and add no `vehicles` cost; they are charged distance and
  time like any trip.
- A request is `SERVED` once nothing is left to deliver (delivered or outsourced). Partially
  delivered requests stay `ASSIGNED` while the vehicle restocks.
- **next_node at a recourse epoch**: action 0 (depot) = `return_to_depot`; action = the failing
  customer = `outsource`; the mask allows only the rule's `options`.
- Outsourcing costs are rule parameters (`per_customer`, `per_unit`, numbers or instance
  references). The `outsourcing` constraint in the spec keeps its placeholder meaning until the
  VRP-SDO model is fixed at Gate 2; the provisional `VRPSD-Outsource-v0` references the instance's
  `outsourcing_costs` through the rule.

## Custom rules
Subclass `RecourseRule`, register with `register_recourse(name, params=Model)` (or a package entry
point), and name the rule in YAML. The spec hash covers the name and parameters, not the code. The
validator checks the physics of any rule (quantities, loads, restocks at the depot, coming back to the
same customer, outsourcing cost); for the built-in rules it also checks the rule's behavior.

## Verification
- Exact expectation: on 5 customers with categorical demands (1,024 scenarios), the environment's
  probability-weighted cost of an a priori route equals an independent dynamic program to 1e-9 for
  `return_to_depot` with both values of `partial_service` and `on_exact_depletion`, and for
  `outsource` with both values of `partial_service`.
- Monte Carlo with Poisson demands: 4,000 episodes within the 99.9% confidence interval of the exact
  value.
- Property tests over random stochastic specs and rules: invariants, validator, determinism and
  non-anticipativity.

## Alternatives
- Charging service time once per customer (on the first visit): rejected because a revisit is a
  real stop; either is defensible, and it is written down here.
- A separate action space for recourse epochs: would change every environment's action space;
  reusing node actions keeps `Discrete(max_nodes)`.
