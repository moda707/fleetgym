<!--
Transcribed verbatim from the design document attached to the build request (2026-11-20).
The two embedded diagrams (architecture, roadmap) could not be transcribed; placeholders mark them.
The project is published as FleetGym (`fleetgym`); see docs/adr/0001-package-name.md.
"RouteGym", `routegym` and `rg.` in this document refer to the same library.
-->

# RouteGym: Design Plan for an Open-Source RL Library for Vehicle Routing

Nov 20, 2026 · @Mohsen Dastpak · Working name, availability to be checked

## 1. Vision, scope and non-goals

RouteGym aims to be the standard environment library for sequential decision-making in vehicle routing, the way Gymnasium is for control and games. It targets stochastic and dynamic VRPs first, where information arrives during the day and decisions are made before the future is known.

**Goals**

- One composable problem specification covering the classical VRP family and its stochastic, dynamic, multi-agent and electric extensions.
- Gymnasium and PettingZoo APIs, so any RL library trains on it without adapters.
- Uncertainty as a first-class object: every parameter is fixed or drawn from a user-chosen distribution, with an explicit revelation time.
- Batteries included: default SB3 agents, neural network base classes, OR baselines, visualization and testing tools.
- One reproducible evaluation protocol that compares RL and OR policies on identical random scenarios.

**Primary users**

- RL researchers who want realistic routing benchmarks beyond static solution construction.
- OR researchers who want to test learned policies against strong heuristics, or embed learning in their own algorithms.
- Practitioners prototyping dispatching and charging policies before deployment.

**Non-goals for v1**

- Being a state-of-the-art static solver. PyVRP, HGS and OR-Tools fill that role, and RouteGym calls them.
- Replacing RL4CO for static neural construction. RouteGym interoperates with it.
- Microscopic traffic simulation. Travel times come from models or matrices, not from a simulator such as SUMO.
- Adjacent problems such as 3D loading or crew scheduling.

## 2. Landscape and positioning

Existing libraries cover static solution construction well. Stochastic dynamic routing, strong OR baselines and a shared evaluation protocol remain largely open, and that is where RouteGym sits.

| Library | Backend and API | Strength | Gap RouteGym fills |
| --- | --- | --- | --- |
| [RL4CO](https://github.com/ai4co/rl4co) | PyTorch, TorchRL, TensorDict | Many static routing envs, models and trainers on GPU | Little stochasticity, no time-driven dynamics |
| [RouteFinder](https://github.com/ai4co/routefinder) | Built on RL4CO | One attribute-based env covering 24 variants | Static only |
| [Jumanji](https://github.com/instadeepai/jumanji) | JAX, jit-compiled | Very fast TSP, CVRP, MultiCVRP | Narrow routing coverage |
| [MAEnvs4VRP](https://github.com/ricgama/maenvs4vrp) | PyTorch, PettingZoo-style AEC | 13 multi-agent envs, including dynamic and stochastic VRPTW | No shared evaluation protocol, OR baselines or EV module |
| [EURO-NeurIPS 2022](https://github.com/ortec/euro-neurips-vrp-2022-quickstart) | Custom | Dynamic VRPTW with an HGS baseline | One variant, frozen after the competition |
| [OR-Gym](https://github.com/hubbs5/or-gym) | Legacy Gym, Ray 1.0 | Broad OR coverage | Unmaintained |
| [SVRPBench](https://github.com/yehias21/vrp-benchmarks) | Dataset and evaluation scripts | Realistic stochastic urban instances up to 1,000 customers | Not an interactive environment |
| [Routing Arena](https://arxiv.org/abs/2310.04140) | Evaluation suite | Runtime-aware comparison of neural and OR solvers | Static CVRP only |

RouteGym complements these projects rather than competing with them. It ships adapters so RL4CO policies can act as inner solvers, PyVRP supplies OR baselines, and SVRPBench and EURO-NeurIPS data load as instances.

## 3. Design principles

Eight principles decide every design trade-off below.

1. **Model first, code second.** The code mirrors the five elements of a sequential decision problem (state, decision, exogenous information, transition, objective) from Powell (2019, 2022), and the route-based view of Ulmer et al. (2020).
2. **Compose, don't enumerate.** A variant is a set of attributes on one general problem, as in Vidal et al. (2014) and Berto et al. (2025b). New variants rarely need new classes.
3. **Uncertainty is data, not code.** Distributions, revelation times and correlations are declared in the spec and swapped without touching the simulator.
4. **Same scenario, every policy.** Exogenous randomness is pre-sampled into scenarios, so policies are compared with common random numbers.
5. **Standard interfaces only.** Gymnasium, PettingZoo, SB3 and TorchRL work out of the box. No custom training loop is required.
6. **Correctness is testable.** An independent validator re-checks every trajectory, and invariants are property-tested.
7. **Readable, then fast.** A clear Python reference implementation comes first. Hot paths move to numba or C++ behind the same API.
8. **One file reproduces a result.** Every environment is rebuilt from one YAML spec plus a seed, and every result stores the spec hash.

## 4. Architecture overview

RouteGym has six layers, and each depends only on the layers below it. The spec describes a problem, the scenario layer samples its randomness, the simulator advances time, and everything an agent touches sits above the simulator.

[Embedded diagram not transcribed: RouteGym architecture · 6 layers, external libraries on the right]

Read from the bottom up: a spec becomes scenarios, the simulator replays them, and agents and tools only ever see the interface layer.

**Package layout**

```text
routegym/
  spec/          problem specification, schema, YAML loader, env registry
  uncertainty/   distributions, arrival processes, revelation, correlation, scenarios
  network/       Euclidean, matrix, OSM/OSRM, time-dependent travel models
  constraints/   capacity, time windows, duration, energy, precedence, ...
  ev/            energy models, batteries, charging curves, stations, queues
  sim/           event engine, state, transitions, commitment, recourse
  interfaces/    action interfaces, observation builders, rewards, masks
  envs/          Gymnasium envs, PettingZoo envs, wrappers, vector envs
  baselines/     greedy, insertion, PyVRP re-optimization, rollout, oracle
  agents/        SB3 defaults, policies, MARL trainers
  nn/            base classes, encoders, decoders, built-in architectures
  data/          instance readers, generators, dataset downloads
  viz/           renderers, replay viewer, dashboards, logging callbacks
  testing/       validator, invariants, property tests, module checker
  eval/          evaluation harness, metrics, statistics, leaderboard export
  cli/           routegym train | eval | render | validate | bench
```

## 5. Problem specification and configuration

A problem is one immutable `ProblemSpec`, built in Python or loaded from YAML and validated against a schema (pydantic) before any simulation runs.

| Component | Contents |
| --- | --- |
| Network | Geometry, distance and time model, time dependence (§8) |
| Nodes | Depots, customers, charging stations and their static attributes |
| Fleet | Vehicle types, counts, sizing mode, start and end depots, shifts |
| Constraints | List of constraint modules with parameters (§9) |
| Uncertainty | One `Uncertain` declaration per random parameter (§6) |
| Dynamics | Request arrival processes, cutoff time, cancellations (§7) |
| Decisions | Decision-epoch rule, action interface, commitment rules, recourse (§11, §12) |
| Objective | Weighted cost terms or a vector of objectives (§14) |
| Agents | Single or multi-agent, reward sharing, observation scope (§15) |

```python
import routegym as rg
from routegym import dist

spec = rg.ProblemSpec(
    network=rg.Euclidean(time_model=rg.SpeedProfile.ichoua_default()),
    fleet=rg.Fleet(types=[rg.VehicleType("van", capacity=100, count=6)],
                   sizing="fixed", shift=(0, 480)),
    constraints=[rg.Capacity(), rg.TimeWindows(soft=True, late_penalty=2.0),
                 rg.RouteDuration(max_minutes=480)],
    uncertainty={
        "demand": rg.Uncertain(dist.Gamma(shape=2, mean="instance"), revealed="on_arrival"),
        "service_time": rg.Uncertain(dist.LogNormal(mu=2.3, sigma=0.4), revealed="on_completion"),
        "travel_time": rg.Uncertain(dist.LogNormalMultiplier(sigma=0.2), revealed="on_completion"),
    },
    dynamics=rg.Dynamics(arrivals=rg.NHPP(rate=rg.Profile.morning_peak(base=0.3)),
                         cutoff=360, cancellations=rg.Bernoulli(p=0.02)),
    decisions=rg.Decisions(epochs="on_event", interface="next_node",
                           commitment="next_stop", recourse="agent"),
    objective=rg.Objective(distance=1.0, lateness=2.0, unserved=50.0),
)
env = rg.make_env(spec, seed=0)
```

- **Registry.** Presets are registered IDs, `rg.make("DVRPTW-StochReq-v0", **overrides)`, mirroring `gymnasium.make`.
- **Versioning.** An env ID's version number changes whenever its dynamics or defaults change, independent of the package version.
- **Reproducibility.** Each spec has a stable hash, saved alongside every log, model checkpoint and result.
- **Overrides.** Any field can be changed from the command line through Hydra, for example `fleet.types.0.count=8`.
- **Extensibility.** Constraints, distributions, energy models and network models are registered plugins, so third-party packages can add their own.

### 5.1 One file per VRP variant

Every VRP in the catalog (§16) is exactly one YAML file, and loading that file is all it takes to get a working environment. A variant that combines existing modules needs no Python code. Genuinely new mechanics, such as a new constraint type, are added once as a plugin and are then available to every config.

**File layout.** The path mirrors the env ID, and the registry is built by scanning this folder.

```text
configs/
  deterministic/   CVRP-v0.yaml  VRPTW-v0.yaml  PDPTW-v0.yaml  ...
  stochastic/      VRPSD-v0.yaml  VRPSD-Outsource-v0.yaml  ...
  dynamic/         DVRP-StochReq-v0.yaml  SDD-v0.yaml  ...
  electric/        EVRPTW-v0.yaml  EVRP-NL-v0.yaml  ...
  multi_agent/     MA-DVRPTW-v0.yaml  ...
```

**Loading.** `rg.make("VRPTW-v0")`, `rg.make("path/to/my_variant.yaml")` and `routegym train --config my_variant.yaml` all accept the same file.

**How a file stays short**

- `extends:` inherits from another config and overrides only what differs. Maps merge deeply, constraints merge by name, and `null` removes an inherited entry.
- `from_instance` takes a value from the instance file, so one config serves a whole benchmark set.
- `meta:` holds the variant's name, defining papers, instance source and expected results (§5.2). It documents the variant and drives the replication suite.
- A JSON Schema generated from the pydantic models gives autocompletion and inline errors in editors such as VS Code.
- `routegym show <config>` prints the fully resolved file after inheritance. `routegym validate <config>` checks it without running anything.
- Third-party packages register their own config folders through Python entry points, so others publish variants without touching RouteGym.

**Examples.** Each file below is complete. Parameter values are illustrative.

```yaml
# configs/deterministic/CVRP-v0.yaml
id: CVRP-v0
meta:
  name: Capacitated vehicle routing problem
  references: [dantzig1959]
  instances: {source: cvrplib, set: X}
network: {type: euclidean, rounding: nearest_int}
fleet:
  sizing: unlimited
  types: [{name: truck, capacity: from_instance}]
constraints: {capacity: {}}
objective: {distance: 1.0}
decisions: {interface: next_node}
```

```yaml
# configs/deterministic/VRPTW-v0.yaml
id: VRPTW-v0
extends: CVRP-v0
meta:
  name: VRP with time windows
  references: [solomon1987]
  instances: {source: solomon}
network: {type: euclidean, rounding: none}
constraints:
  time_windows: {soft: false, waiting: allowed}
  route_duration: {max: depot_due_time}
objective: {lexicographic: [vehicles, distance]}
```

```yaml
# configs/stochastic/VRPSD-Outsource-v0.yaml
id: VRPSD-Outsource-v0
extends: CVRP-v0
meta:
  name: Stochastic demands with outsourcing
  references: [vrpsdo_under_review]
fleet:
  sizing: fixed
  types: [{name: truck, capacity: from_instance, count: from_instance}]
uncertainty:
  demand: {dist: poisson, mean: from_instance, revealed: on_arrival}
constraints:
  outsourcing: {cost: per_customer, value: from_instance}
recourse: {capacity_failure: agent_decides}   # or return_to_depot | outsource | split
decisions: {epochs: on_event, interface: next_node}
objective: {distance: 1.0, outsourcing: 1.0}
```

```yaml
# configs/dynamic/DVRP-StochReq-v0.yaml
id: DVRP-StochReq-v0
extends: VRPTW-v0
meta:
  name: Dynamic VRP with stochastic requests
  references: [bent2004, hvattum2006]
dynamics:
  arrivals:
    process: nhpp
    profile: {breaks: [0, 120, 300, 480], rates: [0.2, 0.5, 0.1]}   # requests per minute
  request_class:
    location: {dist: kde, data: from_instance}
    demand: {dist: categorical, values: [5, 10, 20], probs: [0.5, 0.3, 0.2]}
    time_window_width: {dist: uniform, low: 60, high: 180}
  cutoff: 360
  cancellations: {dist: bernoulli, p: 0.02}
decisions: {epochs: on_event, interface: accept_reject, routing: cheapest_insertion}
objective: {distance: 1.0, rejected: 50.0}
```

```yaml
# configs/electric/EVRP-NL-v0.yaml
id: EVRP-NL-v0
extends: CVRP-v0
meta:
  name: EV routing with nonlinear charging functions
  references: [montoya2017, froger2019]
  instances: {source: montoya2017}
ev:
  energy: {model: linear, rate: from_instance}
  battery: {capacity: from_instance, initial: full}
  charging: {function: piecewise_linear, breakpoints: from_instance, partial: true}
  stations: {nodes: from_instance, ports: unlimited}
constraints:
  energy: {}
  route_duration: {max: from_instance}
objective: {time: 1.0}
```

```yaml
# configs/multi_agent/MA-DVRPTW-v0.yaml
id: MA-DVRPTW-v0
extends: DVRPTW-v0
agents: {mode: multi, api: aec, reward: global, conflict: priority}
```

**A variant nobody has published.** Composition is the point: a user can combine dynamic requests, stochastic travel times and an electric fleet in a dozen lines, with no new code.

```yaml
# my_project/DSEVRP-Traffic.yaml
id: DSEVRP-Traffic-v0
extends: EVRP-NL-v0
network:
  time_model: {type: speed_profile, profile: urban_peaks, fifo: true}
uncertainty:
  travel_time: {dist: lognormal_multiplier, sigma: 0.25, revealed: on_completion}
  station_wait: {dist: exponential, mean: 12, revealed: on_arrival}
dynamics:
  arrivals: {process: poisson, rate: 0.3}
  cutoff: 300
decisions: {epochs: on_event, interface: next_node}
```

### 5.2 Replication suite

Each config also has to prove it reproduces the problem from its defining paper. That evidence is what lets the library claim it replicates many VRPs, and it gives contributors a clear bar for adding new ones.

**The `expected` block.** A config's `meta` lists the instances to check and the published values to match, with a tolerance.

```yaml
meta:
  references: [solomon1987]
  instances: {source: solomon, subset: [C101, R101, RC101]}
  expected:
    model_fidelity: {solutions: bks_files, tolerance: 1e-6}
    baseline: {policy: pyvrp, time_limit_s: 60, max_gap: 0.01}
```

**Three levels of evidence**

1. **Model fidelity (deterministic variants).** Published best-known solutions are replayed through the environment as action sequences. Every one must be feasible, and the environment's cost must equal the published cost. This proves the constraints and cost model match the paper, independent of any solver.
2. **Exact expectations (stochastic variants).** Where the literature gives closed-form or exactly computable values, the environment must match them on small instances. One example is the expected cost of an a priori route under the classical return-to-depot recourse (Bertsimas 1992).
3. **Statistical fidelity (dynamic variants).** Generated scenarios must reproduce the published instance statistics, such as arrival rates per period and degree of dynamism, within confidence intervals. Where a paper reports policy results, a reference policy is rerun and compared.

**Running it.** `routegym replicate VRPTW-v0` checks one config, and `routegym replicate --all` checks the whole catalog. CI runs the fast subset on every commit and the full suite before each release.

**Coverage report.** Each release publishes a table of every registered variant with its level of evidence, pass or fail, and the baseline gap. That table goes straight into the software paper as evidence of breadth. New contributions are accepted once their config passes at least one level.

## 6. Uncertainty framework

Every uncertain parameter is declared as `Uncertain(distribution, revealed, correlated_with)`. The same instance becomes deterministic, stochastic or dynamic by editing the spec only.

**Supported uncertainty sources**

| Source | Typical models | Revealed | Defining references |
| --- | --- | --- | --- |
| Customer presence | Bernoulli per customer | At start or on request | Bertsimas (1988); Gendreau et al. (1995) |
| Customer location | Fixed, uniform, clustered, KDE from data, zone mixture | On request | Bent and Van Hentenryck (2004) |
| Request time | Poisson, nonhomogeneous Poisson, Hawkes, log replay | On request | §7 |
| Demand | Poisson, truncated normal, gamma, discrete empirical | On arrival or on request | Bertsimas (1992); Gendreau et al. (1996) |
| Travel time | Lognormal multiplier on a time-dependent mean, incidents | On departure or on completion | Laporte et al. (1992); Kenyon and Morton (2003); Heakl et al. (2025) |
| Service time | Lognormal, gamma, empirical | On completion | Laporte et al. (1992) |
| Time windows | Sampled per request class | On request | Heakl et al. (2025) |
| Energy consumption | Noise on the energy model | On completion | Basso et al. (2022) |
| Charging station wait | Exponential, queue model | On arrival at station | Keskin et al. (2019); Kullman et al. (2021) |
| Vehicle breakdown | Hazard rate per vehicle | When it happens | Mu et al. (2011) |
| Cancellation | Bernoulli or hazard after request | When it happens | Pillac et al. (2013) |

**Distribution API**

- Built-ins: `Fixed`, `Uniform`, `Normal`, `TruncatedNormal`, `LogNormal`, `Gamma`, `Weibull`, `Exponential`, `Poisson`, `NegativeBinomial`, `Bernoulli`, `Categorical`, `Empirical`, `KDE`, `Mixture`.
- Any frozen `scipy.stats` distribution, or any object with `sample(rng, size)`, also works.
- Optional `mean()` and `quantile(q)` let OR baselines plan with expected values or chance constraints.
- Parameters can be functions of time of day, zone, customer class or a latent factor, which gives conditional distributions.
- `Fixed(value)` is the deterministic case. A deterministic instance is the same spec with every source fixed.

**Revelation model**

- Options: `never`, `a_priori` (known at reset), `on_request`, `on_departure`, `on_arrival`, `on_completion`, or a custom event.
- Forecasts are optional. Before the realization, the agent can see a forecast distribution or a noisy point estimate, controlled by `forecast_noise`.
- This turns partial observability into a spec setting rather than a separate environment.

**Correlation and non-stationarity**

- Latent factors (weather, day type, congestion regime) are sampled once per episode and shift several parameters at once.
- Zone-level shocks give spatially correlated travel times.
- Markov-modulated rates switch regimes during the day.
- Copulas for arbitrary joint marginals come in a later release.

**Scenarios and common random numbers**

- A `Scenario` stores every exogenous realization of one episode: arrivals, demands, durations, waits, breakdowns.
- Scenarios are generated from (spec, seed), saved as Parquet, and replayed exactly.
- Each source draws from its own RNG stream, so changing one source leaves the others untouched.
- Evaluation uses fixed scenario sets, so cost differences reflect policies only.
- The full scenario also feeds the perfect-information oracle (§18).

## 7. Dynamic requests: customers arising during the day

Customers can appear at any time during the operating day. Each new request carries its own sampled location, demand, time window and service time, drawn through the uncertainty framework.

**Arrival processes**

- Homogeneous Poisson with rate λ.
- Nonhomogeneous Poisson with a time-of-day profile (piecewise constant or spline), sampled by thinning.
- Spatio-temporal rates per zone and period, or a fitted point-process model.
- Self-exciting (Hawkes) processes for clustered bursts.
- Replay of historical request logs, such as the Amazon Last Mile data (Merchán et al. 2024) or EURO-NeurIPS 2022 (Kool et al. 2023).
- Known-customer mode: a fixed customer set, each requesting with some probability at a random time (Thomas 2007; Ulmer et al. 2018).

**Request classes.** Attributes are sampled jointly through a `RequestClass`, for example residential versus commercial, or small versus bulky. Each class has its own distributions for location, demand, window width and service time, as in SVRPBench (Heakl et al. 2025).

**Dynamism knobs**

- Degree of dynamism (share of requests revealed after the start) and effective degree of dynamism, which weights how late they arrive (Lund et al. 1996; Larsen et al. 2002). Generators can target a given value.
- Advance versus immediate requests, with configurable notice time.
- A cutoff time after which no new requests are accepted.
- Cancellations, and customer patience: a request is abandoned if not confirmed or served by a deadline.

**Decisions specific to dynamic requests**

- Accept, reject or outsource each request on arrival, with or without immediate confirmation (Ulmer et al. 2018; Côté and Potvin 2009).
- Assign a request to a vehicle now, or defer the assignment.
- Wait at the current location or reposition idle vehicles (Mitrović-Minić and Laporte 2004; Branke et al. 2005).
- Return preemptively to the depot to pick up new orders in same-day delivery (Voccia et al. 2019).
- Divert a vehicle already en route when diversion is allowed (Ichoua et al. 2000).

**Reference problems.** Dynamic VRP (Psaraftis 1988; Pillac et al. 2013; Psaraftis et al. 2016), DVRP with stochastic requests (Bent and Van Hentenryck 2004; Hvattum et al. 2006; Ichoua et al. 2006), same-day delivery (Voccia et al. 2019), dynamic dial-a-ride (Cordeau and Laporte 2007), and dynamic stochastic EV routing (Basso et al. 2022). Surveys: Ritzinger et al. (2016) and Soeffker et al. (2022).

## 8. Network, travel time and time dependence

Travel happens on a graph whose arc times can depend on departure time, and every time-dependent model preserves the FIFO property: leaving later never arrives earlier.

**Geometry**

- Euclidean coordinates in a unit square or in kilometers.
- Explicit distance and time matrices, asymmetric allowed.
- Road networks from OpenStreetMap with OSRM or Valhalla matrices, precomputed and cached per instance.

**Travel time models**

| Model | Description | Reference |
| --- | --- | --- |
| Constant speed | Time equals distance over speed | Classical |
| Speed profile | Piecewise-constant speeds per period and arc class; FIFO by construction | Ichoua et al. (2003) |
| Travel time function | Piecewise-linear time per arc, given directly | Malandraki and Daskin (1992) |
| Stochastic time-dependent | Random multiplier on the time-dependent mean, optionally correlated by zone | Kenyon and Morton (2003); Heakl et al. (2025) |
| Incidents | Random disruptions that slow or close arcs or zones for a period | Heakl et al. (2025) |

- **FIFO check.** The network module verifies FIFO on every profile and refuses or repairs functions that break it. Sampling independent travel times per period breaks FIFO, so RouteGym samples speed shocks instead.
- **Arc costs.** Distance, time, energy, tolls and emissions can each carry a weight (Bektaş and Laporte 2011).
- **Review.** Gendreau, Ghiani and Guerriero (2015) survey time-dependent routing.

## 9. Constraint and attribute library

Every constraint implements one interface, so constraints combine freely and new ones are added without changing the simulator.

```python
class Constraint(ABC):
    name: str
    def init_state(self, spec, scenario) -> ConstraintState: ...
    def mask(self, state, candidates) -> np.ndarray: ...       # actions that stay feasible
    def check(self, state, action) -> Violation | None: ...    # hard feasibility
    def update(self, state, event) -> None: ...                # state after an event
    def penalty(self, state, event) -> float: ...              # cost of soft violations
    def lookahead(self, state, route) -> bool: ...             # optional, for plan-level actions
```

Masks use the information known at decision time. Under uncertainty, an unmasked action can still fail later, and recourse handles that case (§11). Every constraint can run hard (masked, with a penalty if violated anyway) or soft (a cost in the reward).

| Constraint | Parameters | Defining reference |
| --- | --- | --- |
| Vehicle capacity, single or multi-dimensional | Capacity per vehicle type | Dantzig and Ramser (1959) |
| Time windows, hard or soft, single or multiple | Penalties, waiting allowed | Solomon (1987) |
| Route duration and shift length | Maximum duration, shift start and end | Christofides et al. (1979) |
| Route length | Maximum distance | Christofides et al. (1979) |
| Fleet size | Fixed, limited, unlimited, or rentable at a cost | Golden et al. (1984) |
| Heterogeneous fleet | Capacity, speed, cost and energy per type | Golden et al. (1984); Li et al. (2021) |
| Multiple depots | Depot capacities, return-to-origin rule | Cordeau et al. (1997) |
| Open routes | No return to the depot | Sariklis and Powell (2000) |
| Multiple trips | Reloads at the depot within a shift | Brandão and Mercer (1997) |
| Pickup and delivery | Pairing, precedence, optional LIFO | Dumas et al. (1991); Li and Lim (2001) |
| Backhauls | Linehauls before backhauls | Goetschalckx and Jacobs-Blecha (1989) |
| Split delivery | Allowed splits per customer | Dror and Trudeau (1989) |
| Ride time and maximum wait | Per request, for passenger transport | Cordeau and Laporte (2007) |
| Driver breaks and working hours | Rule sets per region | Goel (2009) |
| Multiple compartments | Compartment capacities, product compatibility | El Fallahi et al. (2008) |
| Site dependency | Allowed vehicle–customer pairs | Classical |
| Energy and battery | See §10 | Schneider et al. (2014) |
| Profits and optional customers | Prizes, minimum collected prize | Balas (1989); Chao et al. (1996) |
| Outsourcing | Cost per request or per customer | Côté and Potvin (2009) |
| Consistency | Same driver and similar visit times across days | Groër et al. (2009) |
| Workload balance | Maximum spread of route lengths or durations | Matl et al. (2018) |

## 10. Electric vehicles and charging infrastructure

The EV module adds battery state, energy consumption, charging stations and charging decisions. It works for all-electric fleets and for mixed fleets of electric and combustion vehicles.

| Component | Options | Defining reference |
| --- | --- | --- |
| Energy consumption | Linear in distance; load-dependent; speed and grade; temperature factor; stochastic noise | Bektaş and Laporte (2011); Demir et al. (2011); Goeke and Schneider (2015) |
| Battery | Capacity, minimum and maximum state of charge, initial state | Schneider et al. (2014) |
| Charging function | Linear, or piecewise-linear concave (constant current, constant voltage) | Montoya et al. (2017); Froger et al. (2019) |
| Recharge policy | Full or partial | Keskin and Çatay (2016) |
| Charger types | Power levels, compatibility with vehicle types | Hiermann et al. (2016) |
| Station capacity | Number of ports, queues, reservations | Froger et al. (2022) |
| Station waiting time | Deterministic, time-dependent or stochastic | Keskin et al. (2019); Kullman et al. (2021) |
| Station availability | Public stations that may be occupied or out of service | Kullman et al. (2021) |
| Depot charging | Overnight, or during reloads between trips | Classical |
| Energy price | Flat or time-of-use | Classical |
| Battery degradation | Cost per kWh throughput or depth-of-discharge model | Pelletier et al. (2017) |
| Mixed fleet | Electric and combustion vehicles together | Goeke and Schneider (2015); Hiermann et al. (2016) |

**Decisions added**

- Visit a station, which appears as a node in the `next_node` interface.
- Choose a charger and a charge amount or target state of charge.
- Join, wait in or leave a station queue.

**Energy reachability mask.** A vehicle cannot move to a node from which it could not reach a station or depot on its remaining charge. Minimum-energy distances to the nearest station are precomputed, so the mask stays cheap.

**Uncertainty.** Consumption noise, station waits and station availability use the same `Uncertain` framework as every other source.

**Reference problems.** Green VRP (Erdoğan and Miller-Hooks 2012), EVRPTW (Schneider et al. 2014), electric fleet size and mix (Hiermann et al. 2016), EVRP with nonlinear charging (Montoya et al. 2017), capacitated stations (Froger et al. 2022), public stations (Kullman et al. 2021) and dynamic stochastic EV routing (Basso et al. 2022). Review: Pelletier, Jabali and Laporte (2016).

## 11. Simulator and MDP formulation

The simulator is a discrete-event engine. Simulation time jumps from event to event, and the agent is asked for a decision only at decision epochs.

**Events.** Request arrival, vehicle arrival at a node, service start and end, charging start and end, demand revealed, breakdown, cancellation, timer tick, end of horizon.

**Decision epochs**

- On event: whenever a vehicle becomes idle or a request arrives.
- Periodic: every Δ minutes, as in EURO-NeurIPS 2022.
- Hybrid: periodic, plus selected events.
- Events that need no decision are processed internally, without returning control to the agent.

**MDP elements**, following Powell (2022) and Ulmer et al. (2020):

- **State** S\_k: time; vehicle states (position or arc and progress, load, battery, planned route, committed prefix); request states (pending, accepted, assigned, served, rejected, outsourced, cancelled); revealed information and forecasts.
- **Decision** x\_k: defined by the action interface (§12).
- **Exogenous information** W\_{k+1}: read from the scenario.
- **Transition** S\_{k+1} = S^M(S\_k, x\_k, W\_{k+1}).
- **Objective**: minimize the expected total cost, or a risk measure of it.

```latex
\min_{\pi} \; \mathbb{E}\left[ \sum_{k=0}^{K} C\big(S_k, X^{\pi}(S_k)\big) \,\middle|\, S_0 \right]
```

**Route-based state.** The state can hold a planned route per vehicle. Decisions revise the plan, and only its next segment is committed (Ulmer et al. 2020).

**Commitment rules**

- Next-stop commitment: a vehicle commits only to its next destination.
- Diversion: an en-route vehicle can be redirected (Ichoua et al. 2000).
- Acceptance commitment: an accepted request must be served, or outsourced at a penalty.
- Communication delay: decisions reach vehicles after a configurable lag.

**Recourse when uncertainty causes a failure**

- Route failure under stochastic demand: return to the depot and resume (Dror et al. 1989), restock preventively (Yang et al. 2000), split the service, or outsource.
- Missed time window: serve late at a penalty, or skip.
- Battery shortfall: emergency charging at the nearest station at a penalty, or a towing cost.
- Each recourse runs either as a fixed environment rule (a priori optimization) or as an agent decision (dynamic setting).

**Static construction as a special case.** With no uncertainty and the `next_node` interface, the environment reduces to the standard construction MDP of neural combinatorial optimization, so RL4CO-style methods run unchanged.

**Snapshot and restore.** `env.get_state()`, `env.set_state()` and `env.clone()` copy simulator state and RNG streams. Rollout, Monte Carlo tree search and scenario-based baselines need this, and Gymnasium does not standardize it.

## 12. Action interfaces

Several action interfaces run on the same simulator. The same problem can then be studied as node selection, request acceptance, plan revision or heuristic control, without a new environment.

| Interface | Agent chooses | Gymnasium space | Typical methods | Reference |
| --- | --- | --- | --- | --- |
| `next_node` | Next customer, station or depot for the vehicle that triggered the epoch | Discrete with mask | Attention Model, POMO, DQN | Nazari et al. (2018); Kool et al. (2019) |
| `vehicle_node` | Vehicle and next node jointly | Discrete or MultiDiscrete with mask | Heterogeneous-fleet policies | Li et al. (2021) |
| `accept_reject` | Accept, reject or outsource each new request; routing by a built-in heuristic | Discrete(3) | DQN, PPO | Ulmer et al. (2018) |
| `assign` | Vehicle for each request; sequencing by insertion | Discrete(V+1) | PPO, MARL | Joe and Lau (2020) |
| `plan` | A full set of routes; the env validates it and commits the first segment | Sequence or Graph | Hybrid ML and OR | Ulmer et al. (2020) |
| `operator` | Local-search operator or parameters of an embedded heuristic | Discrete(K) or Box | DQN-guided ILS or LNS | Lu et al. (2020); Hottung and Tierney (2020) |
| `wait_reposition` | Wait, or move an idle vehicle to a zone | Discrete(Z+1) | PPO | Branke et al. (2005) |
| `charge` | Station, charger and target state of charge | MultiDiscrete or Box | PPO, SAC | Kullman et al. (2021) |

- **Composite interfaces.** A `Dict` action combines interfaces, for example `accept_reject` plus `assign`.
- **Invalid actions.** Three modes: mask (default; the mask is in `info["action_mask"]` and in the observation), penalize, or repair to the nearest feasible action. Masking follows Huang and Ontañón (2022).
- **Solver hooks.** The `plan` and `operator` interfaces call a `Solver` protocol (PyVRP, OR-Tools or a custom ILS). Hybrid agents such as ILS guided by a DQN are therefore first-class.

## 13. Observations and state representation

The canonical observation is a typed graph, with converters to padded arrays and flat vectors for simpler agents.

| Element | Features |
| --- | --- |
| Customer node | Coordinates, revealed demand or its forecast, time window, service-time estimate, status, request age, request class |
| Depot or station node | Coordinates, free ports, expected wait, energy price |
| Vehicle | Position or arc and progress, load, battery, clock, planned route, committed segment, type |
| Edge (optional, k-nearest) | Distance, current travel-time estimate, energy |
| Global | Time of day, remaining horizon, pending request count, latent regime if observable, forecast summaries |

- **Formats.** TensorDict (TorchRL and RL4CO), PyG `HeteroData`, a dict of padded arrays with masks (SB3), or a flat vector.
- **Variable size.** Requests appear and disappear, so arrays are padded to `max_nodes` with a validity mask.
- **Scope.** Full, revealed-only, local per agent (radius or k-nearest), with optional observation noise.
- **Wrappers.** `FlattenGraph`, `NormalizeObs`, `History` (recent epochs stacked), `ForecastFeatures`, `RelativeCoordinates`.

## 14. Rewards and objectives

The reward at each epoch is the negative cost incurred since the previous epoch, so the episode return equals the negative total cost by construction.

**Cost terms.** Travel distance, travel time, vehicle fixed cost, rental, lateness, waiting, unserved or rejected requests, outsourcing, energy, charging time, emissions, overtime, workload imbalance, and collected prizes as negative costs. Each term is reported separately in `info["costs"]`.

- **Aggregation.** Weighted sum by default. Vector rewards compatible with MO-Gymnasium (Felten et al. 2023), lexicographic ordering, or a constrained mode that returns constraint costs in `info["cost"]` for safe RL.
- **Shaping.** Potential-based shaping is available for training and always off in benchmarks.
- **Risk.** Evaluation reports CVaR and quantiles. Wrappers support risk-sensitive training.
- **Normalization.** Per-instance scaling, for example dividing by a baseline's cost, keeps rewards comparable across sizes and variants in multi-task training (Berto et al. 2025b).
- **References.** Multi-objective VRPs: Jozefowiez et al. (2008). Workload equity: Matl et al. (2018). Emissions: Bektaş and Laporte (2011).

## 15. Multi-agent reinforcement learning

Each vehicle can be its own agent through PettingZoo's AEC and Parallel APIs (Terry et al. 2021). A separate global state supports centralized training with decentralized execution.

| Issue | RouteGym approach |
| --- | --- |
| Asynchronous decisions | AEC: only agents that need a decision act. Parallel: busy agents receive a no-op and a mask. |
| Conflicts | When two agents pick the same request, resolve by priority, distance or lottery, or reject the joint action at a penalty. |
| Heterogeneous agents | Per-type observation and action spaces, for example electric versus combustion, or van versus truck. |
| Variable agent count | Vehicles join (rentals) or leave (breakdowns, shift end); the agent list updates every step. |
| Observation scope | Local view, optional message channel, and `state()` for a centralized critic, as in SMAC (Samvelyan et al. 2019). |
| Reward | Global, individual, or a mix set by one coefficient, to study credit assignment. |
| Communication | Optional message vectors broadcast each epoch, with configurable delay. |
| Termination | Per-agent truncation at shift end. |

**Training support**

- Built-in MAPPO with parameter sharing (Yu et al. 2022) and IPPO (de Witt et al. 2020).
- Adapters to BenchMARL (Bettini et al. 2024) and TorchRL for QMIX (Rashid et al. 2018), MADDPG (Lowe et al. 2017) and others.
- SB3 has no native multi-agent support. A parameter-sharing wrapper turns homogeneous vehicles into a vectorized single-agent problem, so SB3 defaults still apply.

**Routing-specific references.** Bono et al. (2021) for multi-agent attention routing, and Gama et al. (2025) for the MAEnvs4VRP environment design.

## 16. Variant catalog with defining papers

RouteGym ships 42 registered single-agent presets in four families, each also available as a multi-agent version. Every preset is one YAML config file built from the same modules, not a separate class, and is checked by the replication suite (§5.1, §5.2). Rows marked later are planned after v1.0.

**Deterministic family**

| Env ID | Problem | Defining papers | Instances |
| --- | --- | --- | --- |
| `TSP-v0` | Traveling salesman | Classical | TSPLIB |
| `CVRP-v0` | Capacitated VRP | Dantzig and Ramser (1959) | Uchoa et al. (2017); Christofides et al. (1979) |
| `DCVRP-v0` | CVRP with duration or length limits | Christofides et al. (1979) | Christofides et al. (1979) |
| `VRPTW-v0` | VRP with time windows | Solomon (1987) | Solomon; Gehring and Homberger (1999) |
| `MDVRP-v0` | Multi-depot VRP | Cordeau et al. (1997) | Cordeau et al. (1997) |
| `FSMVRP-v0` | Fleet size and mix, heterogeneous fleet | Golden et al. (1984) | Golden et al. (1984) |
| `OVRP-v0` | Open VRP | Sariklis and Powell (2000) | Generated |
| `MTVRP-v0` | Multi-trip VRP | Brandão and Mercer (1997) | Generated |
| `VRPB-v0` | VRP with backhauls | Goetschalckx and Jacobs-Blecha (1989) | Goetschalckx and Jacobs-Blecha (1989) |
| `SDVRP-v0` | Split delivery VRP | Dror and Trudeau (1989) | Generated |
| `PDPTW-v0` | Pickup and delivery with time windows | Dumas et al. (1991); Li and Lim (2001) | Li and Lim (2001) |
| `DARP-v0` | Dial-a-ride | Cordeau and Laporte (2007) | Cordeau and Laporte (2007) |
| `TOP-v0` | Team orienteering | Chao et al. (1996); Vansteenwegen et al. (2011) | Chao et al. (1996) |
| `PCVRP-v0` | Prize-collecting VRP | Balas (1989) | Generated |
| `VRPPC-v0` | Private fleet and common carrier (outsourcing) | Côté and Potvin (2009) | Côté and Potvin (2009) |
| `TDVRP-v0` | Time-dependent VRP | Malandraki and Daskin (1992); Ichoua et al. (2003) | Ichoua et al. (2003) |
| `MultiAttr-v0` | Mix of attributes for multi-task training | Vidal et al. (2014); Berto et al. (2025b) | Generated |

**Stochastic family (a priori routes with recourse, or dynamic recourse)**

| Env ID | Problem | Uncertainty | Defining papers |
| --- | --- | --- | --- |
| `VRPSD-v0` | Stochastic demands | Demand on arrival | Bertsimas (1992); Gendreau et al. (1996); Secomandi (2001) |
| `VRPSD-Restock-v0` | Stochastic demands with preventive restocking | Demand | Yang et al. (2000) |
| `MVRPSD-DL-v0` | Multi-vehicle stochastic demands with duration limits | Demand | Goodson et al. (2013) |
| `VRPSD-Outsource-v0` | Stochastic demands with outsourcing | Demand | Author's VRP-SDO paper (under revision, Transportation Research Part C) |
| `VRPSC-v0` | Stochastic customers | Presence | Bertsimas (1988); Gendreau et al. (1995) |
| `VRPSTT-v0` | Stochastic travel and service times | Times | Laporte et al. (1992); Kenyon and Morton (2003) |

**Dynamic family**

| Env ID | Problem | Dynamics | Defining papers | Instances |
| --- | --- | --- | --- | --- |
| `DVRP-StochReq-v0` | Dynamic VRP with stochastic requests | Arrivals | Bent and Van Hentenryck (2004); Hvattum et al. (2006) | Kilby et al. (1998) |
| `DVRPTW-v0` | Dynamic VRPTW with periodic epochs | Arrivals | Kool et al. (2023) | EURO-NeurIPS 2022 |
| `DVRP-Accept-v0` | Single vehicle, request acceptance under a time budget | Arrivals | Ulmer et al. (2018) | Generated |
| `SDD-v0` | Same-day delivery with depot returns | Arrivals | Voccia et al. (2019) | Generated |
| `DPDPTW-v0` | Dynamic pickup and delivery | Arrivals | Mitrović-Minić and Laporte (2004) | Li and Lim, dynamized |
| `DDARP-v0` | Dynamic dial-a-ride | Arrivals, cancellations | Cordeau and Laporte (2007) | Generated |
| `DSVRP-Urban-v0` | Dynamic and stochastic urban delivery | Arrivals, time-dependent travel, incidents | Heakl et al. (2025) | SVRPBench |
| `LastMile-v0` | Real last-mile delivery replay | Log replay | Merchán et al. (2024) | Amazon Last Mile |
| `RTVRP-Disrupt-v0` | Real-time rerouting after breakdowns | Breakdowns | Mu et al. (2011) | Generated |

**Electric family**

| Env ID | Problem | Defining papers | Instances |
| --- | --- | --- | --- |
| `GVRP-v0` | Green VRP with refueling stations | Erdoğan and Miller-Hooks (2012) | Erdoğan and Miller-Hooks (2012) |
| `EVRPTW-v0` | EV routing with time windows and recharging stations | Schneider et al. (2014) | Schneider et al. (2014) |
| `EVRPTW-PR-v0` | Partial recharging | Keskin and Çatay (2016) | Schneider et al. (2014) |
| `EFSMVRPTW-v0` | Electric fleet size and mix | Hiermann et al. (2016) | Hiermann et al. (2016) |
| `EVRP-NL-v0` | Nonlinear charging functions | Montoya et al. (2017); Froger et al. (2019) | Montoya et al. (2017) |
| `EVRP-CS-v0` | Capacitated charging stations | Froger et al. (2022) | Generated |
| `EVRP-TDW-v0` | Time-dependent waits at stations | Keskin et al. (2019) | Generated |
| `EVRP-Public-v0` | Public stations with stochastic waits | Kullman et al. (2021) | Generated |
| `DSEVRP-v0` | Dynamic stochastic EV routing | Basso et al. (2022) | Generated |
| `MixedFleet-v0` | Mixed electric and combustion fleet | Goeke and Schneider (2015) | Goeke and Schneider (2015) |

**Multi-agent family.** Every preset above also registers a `MA-` version with one agent per vehicle, for example `MA-DVRPTW-v0` and `MA-EVRPTW-v0` (Bono et al. 2021; Gama et al. 2025).

**Later releases.** Two-echelon VRP (Perboli et al. 2011), truck and drone (Murray and Chu 2015), driver working hours (Goel 2009), consistent VRP (Groër et al. 2009), workload equity (Matl et al. 2018), multi-compartment (El Fallahi et al. 2008).

**Surveys behind the catalog.** Toth and Vigo (2014); Pillac et al. (2013); Psaraftis et al. (2016); Ritzinger et al. (2016); Oyola et al. (2018); Soeffker et al. (2022); Pelletier et al. (2016).

## 17. Instances, generators and datasets

Instances come from classic benchmark files, parametric generators and real data, and all three load into the same `ProblemSpec`.

**Readers**

- VRPLIB and CVRPLIB formats (through the `vrplib` package), TSPLIB.
- Solomon, Gehring and Homberger, Li and Lim, Cordeau MDVRP and DARP sets.
- EV sets of Schneider et al. (2014) and Montoya et al. (2017).
- Large-scale sets of Uchoa et al. (2017) and Arnold et al. (2019).
- Dynamic data: Kilby et al. (1998), EURO-NeurIPS 2022, SVRPBench (Hugging Face), Amazon Last Mile (Merchán et al. 2024).

**Generators**

- Node placement: uniform, clustered or mixed, as in Solomon's R, C and RC classes and the depot and customer positioning of Uchoa et al. (2017).
- Demand, time window (width, tightness), fleet and station placement (grid, k-means, real locations).
- Request processes from §7.
- Explicit distribution-shift knobs: size, spatial pattern, demand scale, degree of dynamism, noise level.

**Data management**

- Train, validation and test splits by seed range and by distribution (in-distribution, shifted, real) for generalization studies.
- Curriculum schedules over size or dynamism.
- A dataset hub on Hugging Face, versioned and checksummed: `routegym.data.download("svrpbench")`.
- Data that cannot be redistributed is fetched by download scripts from its original source.

**Offline RL datasets.** Trajectories from OR baselines are recorded in Minari format (Younis et al. 2024) for imitation learning and offline RL.

## 18. OR baselines and information-relaxation bounds

Every environment ships with OR baselines and a perfect-information bound, so a learned policy is always judged against strong references and against the value of information.

| Baseline | Kind | Setting | Reference |
| --- | --- | --- | --- |
| Random feasible | Sanity check | All | None |
| Nearest neighbor, greedy | Myopic rule | All | Classical |
| Cheapest insertion | Myopic rule | Dynamic | Classical |
| Myopic re-optimization | Re-solve with PyVRP (HGS) at every epoch on known information | Dynamic | Vidal (2022); Wouda et al. (2024) |
| Waiting strategies | Drive-first, wait-first, dynamic waiting | Dynamic | Mitrović-Minić and Laporte (2004); Branke et al. (2005) |
| Multiple scenario approach | Solve sampled futures, pick the consensus decision | Dynamic | Bent and Van Hentenryck (2004) |
| Sample scenario hedging | Scenario-based hedging | Dynamic | Hvattum et al. (2006) |
| Rollout | One-step lookahead with a base heuristic, using `env.clone()` | Stochastic and dynamic | Secomandi (2001); Goodson et al. (2013, 2017) |
| A priori route with recourse | Fixed route evaluated in expectation | Stochastic demand | Bertsimas (1992) |
| Expected-value planning | Plan on means, react with recourse | All | Classical |
| Perfect-information oracle | Solve the full scenario in hindsight | All | Brown et al. (2010) |
| RL4CO policy adapter | Neural construction on static subproblems | Static and epoch-based | Berto et al. (2025a) |

- **One protocol for all.** Baselines implement the same `Policy` protocol as RL agents, `act(obs, info) -> action`, and run through the same evaluation harness.
- **Oracle bound.** For minimization, the hindsight solution is a lower bound on any non-anticipative policy. The gap to it measures how much is left to gain, and how much is the price of uncertainty.
- **Static gap.** For deterministic presets, gaps are also reported against best-known solutions from the instance libraries.

## 19. RL training stack

A user trains a working agent with one command using SB3 defaults (Raffin et al. 2021), then moves to custom networks or other libraries without changing the environment.

**SB3 integration**

- Every environment is SB3-ready through wrappers: dict observations of padded arrays, `ActionMasker` from sb3-contrib, `SubprocVecEnv`, and `VecNormalize` where it is safe.
- Default hyperparameters per env ID live in YAML files (rl-zoo style), tuned with Optuna and published with their results.
- Callbacks: evaluation on fixed scenario sets, logging of each cost component, render snapshots, early stopping, checkpoints pushed to the Hugging Face Hub.

| Algorithm | Package | Use when |
| --- | --- | --- |
| MaskablePPO | sb3-contrib | Default for every discrete, masked interface |
| RecurrentPPO | sb3-contrib | Partial observability, forecasts, history matters |
| MaskableDQN | RouteGym (masked Q-network on SB3 DQN) | `accept_reject`, `operator` selection |
| QR-DQN | sb3-contrib | Risk-aware value learning |
| A2C | SB3 | Quick baselines |
| PPO, SAC, TD3 | SB3 | Continuous interfaces (charge targets, heuristic parameters) |

SB3's DQN has no native action masking. RouteGym's `MaskableDQN` sets masked Q-values to minus infinity in both action selection and the target computation.

```python
import routegym as rg
from routegym.agents import sb3_defaults
from sb3_contrib import MaskablePPO

env = rg.make_vec("VRPSD-Outsource-v0", n_envs=16, interface="next_node")
model = MaskablePPO(
    policy=rg.nn.MaskableGraphPolicy,
    env=env,
    policy_kwargs=dict(features_extractor_class=rg.nn.GATEncoder,
                       features_extractor_kwargs=dict(embed_dim=128, n_layers=3)),
    **sb3_defaults("VRPSD-Outsource-v0", "maskable_ppo"),
)
model.learn(2_000_000, callback=rg.callbacks.ScenarioEval(n_scenarios=256))
```

**Command line.** `routegym train --env DVRPTW-StochReq-v0 --algo maskable_ppo --policy gat --seed 0`, then `routegym eval` and `routegym render` on the saved model.

**Beyond SB3**

- TorchRL and RL4CO adapters for batched GPU training on static and epoch-based subproblems, including REINFORCE with POMO-style shared baselines (Kwon et al. 2020).
- CleanRL-style single-file scripts for readers who want every line visible (Huang et al. 2022).
- Multi-agent trainers as in §15.

## 20. Neural network base classes

A small hierarchy of abstract modules lets users write their own encoder, decoder or full policy, while RouteGym handles masking, batching, padding and SB3 integration.

```python
class RouteGymModule(nn.Module, ABC):       # config, registry, device handling
class ObservationAdapter(ABC):              # observation dict -> GraphBatch (PyG or TensorDict)
class Encoder(RouteGymModule):              # GraphBatch -> node, vehicle and global embeddings
    def forward(self, g: GraphBatch) -> Embeddings: ...
class ContextBuilder(RouteGymModule):       # embeddings + dynamic state -> decoder query
class Decoder(RouteGymModule):              # query + embeddings + mask -> logits
    def forward(self, ctx, emb, mask) -> torch.Tensor: ...
class ValueHead(RouteGymModule):            # embeddings -> V(s) or Q(s, .)
class RouteGymPolicy(MaskableActorCriticPolicy):  # SB3 policy assembled from the parts
class RouteGymQNetwork(QNetwork):           # SB3 DQN-compatible, masked
class MultiAgentPolicy(RouteGymModule):     # shared or per-type actors, centralized critic
```

**Built-in architectures**

| Name | Kind | Reference |
| --- | --- | --- |
| `MLPEncoder` | Flat baseline on flattened observations | Classical |
| `GCNEncoder` | Graph convolution | Kipf and Welling (2017); Joshi et al. (2019) |
| `GATEncoder` | Graph attention with edge features | Veličković et al. (2018) |
| `AttentionModelEncoder` with `PointerDecoder` | Transformer encoder, pointer decoder | Vinyals et al. (2015); Kool et al. (2019) |
| `POMODecoder` | Multi-start decoding with a shared baseline | Kwon et al. (2020) |
| `HeteroGNNEncoder` | Separate customer, depot, station and vehicle node types | Fey and Lenssen (2019) |
| `DynamicContext` | Encodes time, load, battery and pending requests per step | Nazari et al. (2018) |
| `FleetSetEncoder` | Permutation-invariant encoding of the vehicles | Lee et al. (2019) |
| `AttributeEncoder` | Variant attribute embeddings for multi-task training | Berto et al. (2025b) |
| `OperatorSelector` | Q-network over local-search operators from solution features | Author's VRP-SDO design (ILS with DQN and GAT) |

**Customization contract**

- Subclass `Encoder`, `Decoder` or `ContextBuilder`, implement `forward`, and register the class with `@rg.nn.register("my_encoder")`.
- Use it by name in configs or pass the class directly to an SB3 policy.
- `rg.testing.check_module(MyEncoder)` checks output shapes, masked logits equal to minus infinity, permutation equivariance over customers, batch independence and padding invariance.

```python
@rg.nn.register("edge_gat")
class EdgeGAT(rg.nn.Encoder):
    def __init__(self, embed_dim=128, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(GATv2Conv(embed_dim, embed_dim, edge_dim=3)
                                    for _ in range(n_layers))
    def forward(self, g):
        h = self.embed_nodes(g)                 # helper from the base class
        for layer in self.layers:
            h = h + layer(h, g.edge_index, g.edge_attr).relu()
        return self.pack(h, g)                  # returns Embeddings(node, vehicle, global)
```

## 21. Visualization tools

All visualization reads from one event log. Any episode can be rendered live, replayed later or compared side by side without rerunning the simulation.

**Render modes.** `human` (interactive), `rgb_array` (frames for Gymnasium's `RecordVideo`), `ansi` (text summary), and `html` (a self-contained replay file).

| View | Shows |
| --- | --- |
| Map | Depots, customers by status, stations, vehicle positions, planned versus committed route segments, new requests highlighted on arrival |
| Vehicle Gantt | Travel, service, waiting, charging and idle time per vehicle, with time windows overlaid |
| Request timeline | Arrival, acceptance, assignment, service and lateness for every request |
| Load and battery traces | Load and state of charge per vehicle over time, charging events, thresholds |
| Uncertainty overlay | Forecast versus realized demand and travel time; a fan of sampled futures at one decision |
| Cost breakdown | Stacked cost components over time and per episode |
| Policy inspector | Action probabilities or Q-values drawn on the map at a decision, attention weights, the mask |
| Policy comparison | Two policies on the same scenario, side by side with a shared timeline |
| Training dashboard | Learning curves, cost components, gap to baselines (TensorBoard, Weights and Biases) |
| Benchmark report | Tables and plots across scenario sets with confidence intervals |

**Implementation**

- The event log is stored in Arrow or Parquet, one row per event, with the spec hash and scenario ID.
- A web replay viewer ships as one HTML file with a timeline scrubber. It uses Leaflet or deck.gl for real maps and a canvas for Euclidean instances.
- Matplotlib backends produce vector figures for papers.
- Rendering runs headless in CI, so every release regenerates its gallery.

**Efficiency**

- Rendering is decoupled from stepping. Training pays nothing unless logging is on.
- The logging rate is configurable, for example every 100th episode.
- Large instances are downsampled for display, never for computation.

## 22. Testing and validation tools

The testing tools check three things: the environments, the user's agents and the user's custom modules. All of them run in CI on every commit, and users can run them on their own extensions.

| Tool | What it checks |
| --- | --- |
| Env checker | Gymnasium `check_env` and PettingZoo `api_test`, plus RouteGym checks: spaces, mask agreement with the validator, seeding, rendering |
| Solution validator | Independent re-check of every trajectory (capacity, windows, duration, energy, precedence); recomputed cost must equal the cumulative reward |
| Invariant monitors | Runtime assertions in debug mode: no negative load, monotone clock, FIFO, battery within bounds, each request in exactly one state |
| Determinism tests | Same spec and seed give an identical scenario and trajectory across runs, processes and platforms |
| Property-based tests | Hypothesis-generated specs and action sequences never crash, and masks never allow what the validator rejects (MacIver et al. 2019) |
| Golden tests | Tiny instances with known optima from brute force or a MIP, and exact expected costs for small stochastic-demand instances |
| Sampler tests | Kolmogorov–Smirnov and chi-square tests that samplers match declared distributions; arrival-rate checks per period |
| Baseline regression | Baseline costs on fixed benchmark scenarios stay within tolerance across releases |
| Module checker | Shapes, masking, permutation equivariance, batch and padding independence for custom networks (§20) |
| Agent sanity suite | Tiny environments that any correct agent must solve, to catch training bugs early |
| Performance benchmarks | Steps per second, reset time and memory per preset, tracked over time with airspeed velocity |

**CI matrix.** Python 3.10 to 3.13; Linux, macOS and Windows; CPU and GPU runners; documentation build; 90% line coverage target on the core packages.

## 23. Evaluation protocol and leaderboard

A result counts only if it follows one protocol: fixed scenario sets, a decision-time budget, paired statistics and a reported gap to the oracle.

1. **Fixed test sets.** Each env ID has a fixed set of test scenarios, never used for training or tuning.
2. **Common random numbers.** All policies face the same scenarios.
3. **Decision-time budget.** A wall-clock limit per decision epoch on reference hardware, or normalized runtime as in Routing Arena (Thyssens et al. 2023).
4. **Metrics.** Mean cost with a 95% confidence interval, gap to the oracle and to the best baseline, CVaR at 90%, service rate, lateness, share rejected or outsourced, energy, decision time.
5. **Statistics.** Paired t-tests or Wilcoxon signed-rank tests, and performance profiles across instances.
6. **Generalization tracks.** In-distribution, shifted (size, spatial pattern, dynamism, noise) and real data.
7. **Report card.** A JSON file with the spec hash, package versions, seeds, hardware and training budget.

**Leaderboard.** A static site built from submitted report cards. Top entries are verified by rerunning them on hidden scenarios.

## 24. Performance engineering

Speed comes from vectorizing across many environments, not from parallelizing inside one event loop. Performance budgets per preset are set from v0.1 profiling and tracked on every release.

- **Profile first.** A pure-Python reference implementation, then numba kernels for masks, distance and time lookups, and constraint checks.
- **Native core.** Once the API is stable, the event engine moves to C++ behind pybind11 or nanobind, with the Python version kept as the reference for tests.
- **Vectorization.** Asynchronous vector envs, batched resets and shared-memory instance tensors.
- **GPU backend.** For static and periodic-epoch presets, a batched TensorDict backend advances all environments in lockstep, as RL4CO does.
- **Caching.** Precomputed distance, time and energy matrices; k-d trees for neighbor features.
- **Lazy work.** Observations are built only when requested, and masks update incrementally after each event.

## 25. Packaging, documentation, community and governance

RouteGym ships as one permissively licensed package with optional extras, documentation organized around tutorials, and a contribution path designed for new variants.

- **License.** MIT, or Apache-2.0 if an explicit patent grant matters to industrial contributors.
- **Install.** `pip install routegym`, with extras `[sb3]`, `[marl]`, `[ev]`, `[viz]`, `[osm]`, `[solvers]`, `[all]`. Python 3.10 or later.
- **Versioning.** Semantic versioning for the package; env IDs versioned separately (`-v0`, `-v1`).
- **Documentation.** MkDocs or Sphinx with tutorials: first env in five minutes, a custom distribution, a custom constraint, a custom network, multi-agent training, EV routing, running the evaluation protocol. Plus an API reference and a variant gallery with each defining paper.
- **Notebooks.** Every tutorial runs on Colab.
- **Contribution.** Templates for new constraints, distributions and presets; issue labels; a code of conduct; a CITATION.cff file; a Zenodo DOI per release.
- **Governance.** A small maintainer group from the start. Possible homes: the Farama Foundation ecosystem, or affiliation with the AI4CO community.
- **Outreach.** Talks and tutorials at VeRoLog, TRISTAN, INFORMS TSL and NeurIPS workshops; early contact with the PyVRP and RL4CO maintainers.

## 26. Roadmap and milestones

The first usable release, v0.1, is planned for month 5 and centers on stochastic dynamic routing. Each phase ends at a gate that must pass before the next phase starts. The month estimates assume part-time work by one lead maintainer and two contributors.

[Embedded diagram not transcribed: Roadmap · 5 phases, 4 gates, months from project start]

The gates are tests, not dates. A phase that runs long moves the following ones, and nothing ships past a failed gate.

## 27. Risks and mitigations

Scope creep is the largest risk. The mitigation is to build the composition layer first and treat every variant as a preset.

| Risk | Mitigation |
| --- | --- |
| Scope creep | v0.1 limited to the core; variants are presets, not classes |
| Slow Python simulator | Profile early; numba then C++ behind a stable API; GPU backend for static presets |
| Masks disagree with outcomes under uncertainty | Masks use known information only; recourse handles failures; validator and property tests |
| Combinatorial actions fit Gym spaces poorly | Several action interfaces; solver hooks; graph and sequence spaces |
| Overlap with RL4CO and MAEnvs4VRP | Adapters and collaboration; focus on stochastic dynamic routing and the protocol |
| Overfitting to public test sets | Hidden scenarios for leaderboard verification |
| Limited maintainer time | Co-maintainers early; contribution templates; student projects |
| Dependency churn (SB3, Gymnasium, PyG) | Pinned extras; nightly CI against latest releases |
| Data licensing | Redistribute only open data; download scripts for the rest |

## 28. Publication plan

The library supports three publications, and the VRP-SDO paper becomes its first case study.

- **Software paper.** INFORMS Journal on Computing, Software Tools area, where MAEnvs4VRP appeared (Gama et al. 2025). JOSS is a shorter alternative.
- **Benchmark paper.** NeurIPS Datasets and Benchmarks track: the presets, the evaluation protocol, OR and RL baselines, and oracle gaps.
- **Methods study.** A comparison of RL, rollout and scenario-based policies across stochastic dynamic variants, aimed at Transportation Science or Transportation Research Part C.
- **Case study.** Reproducing the VRP-SDO results inside RouteGym validates the library and adds a citable, reusable environment to that paper.

## 29. References

Linked entries were checked online for this plan. The other bibliographic details were compiled from memory and need verification before they appear in a paper.

**Classical and rich VRP variants**

- Balas, E. (1989). The prize collecting traveling salesman problem. *Networks*.
- Brandão, J., Mercer, A. (1997). A tabu search algorithm for the multi-trip vehicle routing and scheduling problem. *European Journal of Operational Research*.
- Chao, I.-M., Golden, B., Wasil, E. (1996). The team orienteering problem. *European Journal of Operational Research*.
- Christofides, N., Mingozzi, A., Toth, P. (1979). The vehicle routing problem. In *Combinatorial Optimization*, Wiley.
- Cordeau, J.-F., Gendreau, M., Laporte, G. (1997). A tabu search heuristic for periodic and multi-depot vehicle routing problems. *Networks*.
- Cordeau, J.-F., Laporte, G. (2007). The dial-a-ride problem: models and algorithms. *Annals of Operations Research*.
- Côté, J.-F., Potvin, J.-Y. (2009). A tabu search heuristic for the vehicle routing problem with private fleet and common carrier. *European Journal of Operational Research*.
- Dantzig, G., Ramser, J. (1959). The truck dispatching problem. *Management Science*.
- Dror, M., Trudeau, P. (1989). Savings by split delivery routing. *Transportation Science*.
- Dumas, Y., Desrosiers, J., Soumis, F. (1991). The pickup and delivery problem with time windows. *European Journal of Operational Research*.
- El Fallahi, A., Prins, C., Wolfler Calvo, R. (2008). A memetic algorithm and a tabu search for the multi-compartment vehicle routing problem. *Computers & Operations Research*.
- Goel, A. (2009). Vehicle scheduling and routing with drivers' working hours. *Transportation Science*.
- Goetschalckx, M., Jacobs-Blecha, C. (1989). The vehicle routing problem with backhauls. *European Journal of Operational Research*.
- Golden, B., Assad, A., Levy, L., Gheysens, F. (1984). The fleet size and mix vehicle routing problem. *Computers & Operations Research*.
- Groër, C., Golden, B., Wasil, E. (2009). The consistent vehicle routing problem. *Manufacturing & Service Operations Management*.
- Li, H., Lim, A. (2001). A metaheuristic for the pickup and delivery problem with time windows. *Proceedings of ICTAI*.
- Matl, P., Hartl, R., Vidal, T. (2018). Workload equity in vehicle routing problems: a survey and analysis. *Transportation Science*.
- Murray, C., Chu, A. (2015). The flying sidekick traveling salesman problem. *Transportation Research Part C*.
- Perboli, G., Tadei, R., Vigo, D. (2011). The two-echelon capacitated vehicle routing problem. *Transportation Science*.
- Sariklis, D., Powell, S. (2000). A heuristic method for the open vehicle routing problem. *Journal of the Operational Research Society*.
- Solomon, M. (1987). Algorithms for the vehicle routing and scheduling problems with time window constraints. *Operations Research*.
- Toth, P., Vigo, D. (eds.) (2014). *Vehicle Routing: Problems, Methods, and Applications*, 2nd ed. SIAM.
- Vansteenwegen, P., Souffriau, W., Van Oudheusden, D. (2011). The orienteering problem: a survey. *European Journal of Operational Research*.
- Vidal, T., Crainic, T., Gendreau, M., Prins, C. (2014). A unified solution framework for multi-attribute vehicle routing problems. *European Journal of Operational Research*.

**Time-dependent routing and emissions**

- Bektaş, T., Laporte, G. (2011). The pollution-routing problem. *Transportation Research Part B*.
- Demir, E., Bektaş, T., Laporte, G. (2011). A comparative analysis of several vehicle emission models for road freight transportation. *Transportation Research Part D*.
- Gendreau, M., Ghiani, G., Guerriero, E. (2015). Time-dependent routing problems: a review. *Computers & Operations Research*.
- Ichoua, S., Gendreau, M., Potvin, J.-Y. (2003). Vehicle dispatching with time-dependent travel times. *European Journal of Operational Research*.
- Malandraki, C., Daskin, M. (1992). Time dependent vehicle routing problems: formulations, properties and heuristic algorithms. *Transportation Science*.

**Stochastic VRPs**

- Bertsimas, D. (1988). *Probabilistic combinatorial optimization problems*. PhD thesis, MIT.
- Bertsimas, D. (1992). A vehicle routing problem with stochastic demand. *Operations Research*.
- Dror, M., Laporte, G., Trudeau, P. (1989). Vehicle routing with stochastic demands: properties and solution frameworks. *Transportation Science*.
- Gendreau, M., Laporte, G., Séguin, R. (1995). An exact algorithm for the vehicle routing problem with stochastic demands and customers. *Transportation Science*.
- Gendreau, M., Laporte, G., Séguin, R. (1996). Stochastic vehicle routing. *European Journal of Operational Research*.
- Goodson, J., Ohlmann, J., Thomas, B. (2013). Rollout policies for dynamic solutions to the multivehicle routing problem with stochastic demand and duration limits. *Operations Research*.
- Kenyon, A., Morton, D. (2003). Stochastic vehicle routing with random travel times. *Transportation Science*.
- Laporte, G., Louveaux, F., Mercure, H. (1992). The vehicle routing problem with stochastic travel times. *Transportation Science*.
- Oyola, J., Arntzen, H., Woodruff, D. (2018). The stochastic vehicle routing problem, a literature review. *EURO Journal on Transportation and Logistics*.
- Secomandi, N. (2001). A rollout policy for the vehicle routing problem with stochastic demands. *Operations Research*.
- Yang, W.-H., Mathur, K., Ballou, R. (2000). Stochastic vehicle routing problem with restocking. *Transportation Science*.

**Dynamic and stochastic dynamic VRPs**

- Bent, R., Van Hentenryck, P. (2004). Scenario-based planning for partially dynamic vehicle routing with stochastic customers. *Operations Research*.
- Branke, J., Middendorf, M., Noeth, G., Dessouky, M. (2005). Waiting strategies for dynamic vehicle routing. *Transportation Science*.
- Brown, D., Smith, J., Sun, P. (2010). Information relaxations and duality in stochastic dynamic programs. *Operations Research*.
- Gendreau, M., Guertin, F., Potvin, J.-Y., Taillard, É. (1999). Parallel tabu search for real-time vehicle routing and dispatching. *Transportation Science*.
- Goodson, J., Thomas, B., Ohlmann, J. (2017). A rollout algorithm framework for heuristic solutions to finite-horizon stochastic dynamic programs. *European Journal of Operational Research*.
- Hildebrandt, F., Thomas, B., Ulmer, M. (2023). Opportunities for reinforcement learning in stochastic dynamic vehicle routing. *Computers & Operations Research*.
- Hvattum, L., Løkketangen, A., Laporte, G. (2006). Solving a dynamic and stochastic vehicle routing problem with a sample scenario hedging heuristic. *Transportation Science*.
- Ichoua, S., Gendreau, M., Potvin, J.-Y. (2000). Diversion issues in real-time vehicle dispatching. *Transportation Science*.
- Ichoua, S., Gendreau, M., Potvin, J.-Y. (2006). Exploiting knowledge about future demands for real-time vehicle dispatching. *Transportation Science*.
- Kilby, P., Prosser, P., Shaw, P. (1998). Dynamic VRPs: a study of scenarios. Technical report, University of Strathclyde.
- Larsen, A., Madsen, O., Solomon, M. (2002). Partially dynamic vehicle routing: models and algorithms. *Journal of the Operational Research Society*.
- Lund, K., Madsen, O., Rygaard, J. (1996). Vehicle routing problems with varying degrees of dynamism. Technical report, DTU.
- Mitrović-Minić, S., Laporte, G. (2004). Waiting strategies for the dynamic pickup and delivery problem with time windows. *Transportation Research Part B*.
- Mu, Q., Fu, Z., Lysgaard, J., Eglese, R. (2011). Disruption management of the vehicle routing problem with vehicle breakdown. *Journal of the Operational Research Society*.
- Pillac, V., Gendreau, M., Guéret, C., Medaglia, A. (2013). A review of dynamic vehicle routing problems. *European Journal of Operational Research*.
- Powell, W. (2019). A unified framework for stochastic optimization. *European Journal of Operational Research*.
- Powell, W. (2022). *Reinforcement Learning and Stochastic Optimization*. Wiley.
- Psaraftis, H. (1988). Dynamic vehicle routing problems. In *Vehicle Routing: Methods and Studies*, North-Holland.
- Psaraftis, H., Wen, M., Kontovas, C. (2016). Dynamic vehicle routing problems: three decades and counting. *Networks*.
- Ritzinger, U., Puchinger, J., Hartl, R. (2016). A survey on dynamic and stochastic vehicle routing problems. *International Journal of Production Research*.
- Soeffker, N., Ulmer, M., Mattfeld, D. (2022). Stochastic dynamic vehicle routing in the light of prescriptive analytics: a review. *European Journal of Operational Research*.
- Thomas, B. (2007). Waiting strategies for anticipating service requests from known customer locations. *Transportation Science*.
- Ulmer, M., Mattfeld, D., Köster, F. (2018). Budgeting time for dynamic vehicle routing with stochastic customer requests. *Transportation Science*.
- Ulmer, M., Goodson, J., Mattfeld, D., Thomas, B. (2020). On modeling stochastic dynamic vehicle routing problems. *EURO Journal on Transportation and Logistics*.
- Voccia, S., Campbell, A., Thomas, B. (2019). The same-day delivery problem for online purchases. *Transportation Science*.

**Electric vehicle routing**

- Basso, R., Kulcsár, B., Sanchez-Diaz, I., Qu, X. (2022). Dynamic stochastic electric vehicle routing with safe reinforcement learning. *Transportation Research Part E*.
- Erdoğan, S., Miller-Hooks, E. (2012). A green vehicle routing problem. *Transportation Research Part E*.
- Froger, A., Mendoza, J., Jabali, O., Laporte, G. (2019). Improved formulations and algorithmic components for the electric vehicle routing problem with nonlinear charging functions. *Computers & Operations Research*.
- Froger, A., Jabali, O., Mendoza, J., Laporte, G. (2022). The electric vehicle routing problem with capacitated charging stations. *Transportation Science*.
- Goeke, D., Schneider, M. (2015). Routing a mixed fleet of electric and conventional vehicles. *European Journal of Operational Research*.
- Hiermann, G., Puchinger, J., Ropke, S., Hartl, R. (2016). The electric fleet size and mix vehicle routing problem with time windows and recharging stations. *European Journal of Operational Research*.
- Keskin, M., Çatay, B. (2016). Partial recharge strategies for the electric vehicle routing problem with time windows. *Transportation Research Part C*.
- Keskin, M., Laporte, G., Çatay, B. (2019). Electric vehicle routing problem with time-dependent waiting times at recharging stations. *Computers & Operations Research*.
- Kullman, N., Goodson, J., Mendoza, J. (2021). Electric vehicle routing with public charging stations. *Transportation Science*.
- Montoya, A., Guéret, C., Mendoza, J., Villegas, J. (2017). The electric vehicle routing problem with nonlinear charging function. *Transportation Research Part B*.
- Pelletier, S., Jabali, O., Laporte, G. (2016). Goods distribution with electric vehicles: review and research perspectives. *Transportation Science*.
- Pelletier, S., Jabali, O., Laporte, G., Veneroni, M. (2017). Battery degradation and behaviour for electric vehicles: review and numerical analyses of several models. *Transportation Research Part B*.
- Schneider, M., Stenger, A., Goeke, D. (2014). The electric vehicle-routing problem with time windows and recharging stations. *Transportation Science*.

**Learning for routing**

- Bello, I., Pham, H., Le, Q., Norouzi, M., Bengio, S. (2017). Neural combinatorial optimization with reinforcement learning. *ICLR Workshop*.
- Berto, F., Hua, C., et al. (2025a). [RL4CO: an extensive reinforcement learning for combinatorial optimization benchmark](https://arxiv.org/abs/2306.17100). *KDD*.
- Berto, F., Hua, C., et al. (2025b). [RouteFinder: towards foundation models for vehicle routing problems](https://arxiv.org/abs/2406.15007). *Transactions on Machine Learning Research*.
- Bono, G., Dibangoye, J., Simonin, O., Matignon, L., Pereyron, F. (2021). Solving multi-agent routing problems using deep attention mechanisms. *IEEE Transactions on Intelligent Transportation Systems*.
- Hottung, A., Tierney, K. (2020). Neural large neighborhood search for the capacitated vehicle routing problem. *ECAI*.
- Joe, W., Lau, H. C. (2020). Deep reinforcement learning approach to solve dynamic vehicle routing problem with stochastic customers. *ICAPS*.
- Joshi, C., Laurent, T., Bresson, X. (2019). An efficient graph convolutional network technique for the travelling salesman problem. *arXiv*.
- Kool, W., van Hoof, H., Welling, M. (2019). Attention, learn to solve routing problems! *ICLR*.
- Kwon, Y.-D., Choo, J., Kim, B., Yoon, I., Gwon, Y., Min, S. (2020). POMO: policy optimization with multiple optima for reinforcement learning. *NeurIPS*.
- Li, J., Xin, L., Cao, Z., Lim, A., Song, W., Zhang, J. (2021). [Deep reinforcement learning for solving the heterogeneous capacitated vehicle routing problem](http://arxiv.org/abs/2110.02629). *IEEE Transactions on Cybernetics*.
- Lu, H., Zhang, X., Yang, S. (2020). A learning-based iterative method for solving vehicle routing problems. *ICLR*.
- Nazari, M., Oroojlooy, A., Snyder, L., Takáč, M. (2018). Reinforcement learning for solving the vehicle routing problem. *NeurIPS*.
- Vinyals, O., Fortunato, M., Jaitly, N. (2015). Pointer networks. *NeurIPS*.

**Multi-agent RL**

- de Witt, C. S., et al. (2020). Is independent learning all you need in the StarCraft multi-agent challenge? *arXiv*.
- Gama, R., Cunha, R., Fuertes, D., del-Blanco, C. R., Fernandes, H. L. (2025). [Multiagent environments for vehicle routing problems](https://pubsonline.informs.org/doi/10.1287/ijoc.2025.1211). *INFORMS Journal on Computing*.
- Lowe, R., et al. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. *NeurIPS*.
- Rashid, T., et al. (2018). QMIX: monotonic value function factorisation for deep multi-agent reinforcement learning. *ICML*.
- Samvelyan, M., et al. (2019). The StarCraft multi-agent challenge. *AAMAS*.
- Terry, J., et al. (2021). PettingZoo: Gym for multi-agent reinforcement learning. *NeurIPS*.
- Yu, C., et al. (2022). The surprising effectiveness of PPO in cooperative multi-agent games. *NeurIPS Datasets and Benchmarks*.

**Neural network building blocks**

- Fey, M., Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *ICLR Workshop*.
- Kipf, T., Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.
- Lee, J., et al. (2019). Set transformer: a framework for attention-based permutation-invariant neural networks. *ICML*.
- Veličković, P., et al. (2018). Graph attention networks. *ICLR*.

**Benchmarks, datasets and software**

- Arnold, F., Gendreau, M., Sörensen, K. (2019). Efficiently solving very large-scale routing problems. *Computers & Operations Research*.
- Bettini, M., Prorok, A., Moens, V. (2024). BenchMARL: benchmarking multi-agent reinforcement learning. *Journal of Machine Learning Research*.
- Felten, F., et al. (2023). A toolkit for reliable benchmarking and research in multi-objective reinforcement learning. *NeurIPS Datasets and Benchmarks*.
- Gehring, H., Homberger, J. (1999). A parallel hybrid evolutionary metaheuristic for the vehicle routing problem with time windows. *EUROGEN*.
- Heakl, A., Shaaban, Y. S., Takáč, M., Lahlou, S., Iklassov, Z. (2025). [SVRPBench: a realistic benchmark for stochastic vehicle routing problem](https://arxiv.org/abs/2505.21887). *NeurIPS Datasets and Benchmarks*.
- Huang, S., Ontañón, S. (2022). A closer look at invalid action masking in policy gradient algorithms. *FLAIRS*.
- Huang, S., et al. (2022). CleanRL: high-quality single-file implementations of deep reinforcement learning algorithms. *Journal of Machine Learning Research*.
- Hubbs, C., et al. (2020). [OR-Gym: a reinforcement learning library for operations research problems](https://arxiv.org/abs/2008.06319). *arXiv*.
- Kool, W., et al. (2023). The EURO Meets NeurIPS 2022 vehicle routing competition. *NeurIPS 2022 Competition Track, PMLR*. [Quickstart repository](https://github.com/ortec/euro-neurips-vrp-2022-quickstart).
- MacIver, D., Hatfield-Dodds, Z., et al. (2019). Hypothesis: a new approach to property-based testing. *Journal of Open Source Software*.
- Merchán, D., et al. (2024). 2021 Amazon Last Mile Routing Research Challenge: data set. *Transportation Science*.
- Raffin, A., et al. (2021). Stable-Baselines3: reliable reinforcement learning implementations. *Journal of Machine Learning Research*.
- Thyssens, D., Dernedde, T., Falkner, J., Schmidt-Thieme, L. (2023). [Routing Arena: a benchmark suite for neural routing solvers](https://arxiv.org/abs/2310.04140). *arXiv*.
- Towers, M., et al. (2024). Gymnasium: a standard interface for reinforcement learning environments. *arXiv*.
- Uchoa, E., et al. (2017). New benchmark instances for the capacitated vehicle routing problem. *European Journal of Operational Research*.
- Vidal, T. (2022). Hybrid genetic search for the CVRP: open-source implementation and SWAP\* neighborhood. *Computers & Operations Research*.
- Wouda, N., Lan, L., Kool, W. (2024). PyVRP: a high-performance VRP solver package. *INFORMS Journal on Computing*.
- Younis, O., et al. (2024). Minari: a dataset API for offline reinforcement learning. *Zenodo / Farama Foundation*.
- [Jumanji repository](https://github.com/instadeepai/jumanji), InstaDeep. [MAEnvs4VRP repository](https://github.com/ricgama/maenvs4vrp).
