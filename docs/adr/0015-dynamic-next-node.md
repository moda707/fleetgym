# 0015: Dynamic requests with next_node (wait, reload, terminate)

Status: accepted (2026-11-20). Decided with the user (plan M6b).

## Context
ADR 0014 handled dynamic requests with request decisions (`accept_reject`, `assign`) and built-in
routing. The user also wants the setting where requests are accepted automatically as they arrive
(until a cutoff, for example the last 20% of the shift) and the policy routes the vehicles with
`next_node`, the set of choosable customers growing during the day, as in their VRP-SDO paper.

## Decision
- Released requests become pending customers at once. Unreleased slots are hidden from observations
  and masks; a cancellation removes a pending request; a vehicle driving to a request that cancels
  arrives, serves nothing and decides again.
- `next_node` under `dynamics` has `Discrete(max_nodes + 1)` actions; the last one is **wait**:
  the vehicle is asked again at the next event (a request arrives or cancels, a vehicle arrives or
  finishes a service; user decision: "until next event") or at its wake-up time, the earlier of the
  end of the arrival window and its latest departure for the depot under the depot's closing time,
  the shift end and the route duration. Waiting is offered only before that time, which depends on
  known information only; whether more requests will actually come is never revealed by a mask.
- **Depot during the day** (user decision): a vehicle returning from a customer is reloaded
  (`restock` row; its route goes on). At the depot it may leave again, wait, or **terminate**
  (choose the depot), terminating only while another vehicle is still working or when nothing is
  left to do (no pending customer and the arrival window is over). A vehicle with nothing feasible
  and nothing left to do ends automatically.
- A vehicle's clock is advanced to the current time when it gets a decision epoch, so masks and
  decisions see the same moment.
- Requirements: `on_event` epochs and a fixed or limited fleet; uncertain parameters and recourse
  together with dynamics are after v0.1.
- Forks redraw the future of the day as in ADR 0014.

## Validation
The validator allows departures and route ends after the vehicle became free (waiting), checks that
vehicles only drive to released requests, and that every released, uncancelled request is served or
reported unserved, besides the release, cancellation and overflow checks of ADR 0014.

## Verification
Unit tests for waiting until an arrival, waking at the end of the arrival window and at the latest
departure, reloading and leaving again, the terminate rule, and cancellations on the way; property
tests over random dynamic problems with all three interfaces (validator, determinism,
non-anticipativity).

## Consequences
- Static configs keep `Discrete(max_nodes)`; dynamic `next_node` configs have one more action.
- Cancellation timing remains an open item (no reference model; user, 2026-11-20).
