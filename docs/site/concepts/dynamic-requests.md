# Requests arriving during the day

```yaml
dynamics:
  arrivals: {process: nhpp, profile: {breaks: [0, 120, 300, 480], rates: [0.2, 0.5, 0.1]}}
  request_class:
    location: {dist: kde, data: from_instance}
    demand: {dist: categorical, values: [5, 10, 20], probs: [0.5, 0.3, 0.2]}
    time_window_width: {dist: uniform, low: 60, high: 180}
  cutoff: 360
  cancellations: {dist: bernoulli, p: 0.02}
  max_requests: 250
```

Arrivals follow a homogeneous Poisson process or a time-of-day rate (thinning), until the cutoff, the
depot's closing time or the shift end. Each request gets a location (`kde`, `uniform` on a square,
or `normal_mixture` clusters), demand, time window and service time. The numbers usually come
from the instance's `distributions` block ({doc}`instances`). The instance's customers are requests
known at the start (`initial_requests: instance`); with `initial_requests: mandatory` they must be
served and are accepted at the start by the routing rule. If they do not fit, `initial_infeasible`
decides: `error` (the default) raises, `serve_all` serves them all late and accepts no other request
that day (Ulmer, Mattfeld and Köster 2017). Unreleased requests are invisible to the agent.

## Two ways to decide

- **Request decisions** (`accept_reject`, `assign`): the agent decides on each request; a built-in
  routing rule (`cheapest_insertion`) drives the vehicles. Accept is masked exactly when no feasible
  insertion exists. `decisions.request_epochs` says when: `on_release` (at once, the default) or
  `on_vehicle_arrival` (at the next arrival of a vehicle, the requests made since then one by one
  in release order, as in Ulmer et al.; at once while a vehicle waits with nothing planned).
- **Routing decisions** (`next_node`): requests are accepted automatically and become customers the
  vehicles can choose. A vehicle may also **wait** until the next event, return to the depot to reload
  and leave again, or end its route at the depot while another vehicle is still working.

Travel times can be rounded up to whole units (`network: {time_rounding: ceil}`), for example
kilometers at 25 km/h in whole minutes (`speed: 0.4166666666666667`).

`degree_of_dynamism` and `effective_degree_of_dynamism` summarize how dynamic a day was.
