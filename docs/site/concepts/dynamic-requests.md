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
depot's closing time or the shift end. Each request gets a location, demand, time window and service
time. The instance's customers are requests known at the start (`initial_requests: instance`).
Unreleased requests are invisible to the agent.

## Two ways to decide

- **Request decisions** (`accept_reject`, `assign`): the agent decides on each request when it
  arrives; a built-in routing rule (`cheapest_insertion`) drives the vehicles. Accept is masked
  exactly when no feasible insertion exists.
- **Routing decisions** (`next_node`): requests are accepted automatically and become customers the
  vehicles can choose. A vehicle may also **wait** until the next event, return to the depot to reload
  and leave again, or end its route at the depot while another vehicle is still working.

`degree_of_dynamism` and `effective_degree_of_dynamism` summarize how dynamic a day was.
