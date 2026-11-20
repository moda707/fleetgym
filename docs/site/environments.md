# Environment catalog

`fleetgym list` prints the registered environments; `fleetgym show <ID>` prints the resolved YAML.

| ID | Problem | Status |
| --- | --- | --- |
| `CVRP-v0` | Capacitated VRP; reproduces CVRPLIB X best-known solutions exactly | stable |
| `VRPTW-v0` | VRP with time windows; reproduces Solomon best-known solutions | stable |
| `VRPSD-v0` | Stochastic demands revealed on arrival, return-to-depot recourse | experimental |
| `DVRP-StochReq-v0` | Requests arriving during the day, accept/reject with cheapest insertion | experimental |
| `MA-CVRP-v0` | Multi-agent CVRP (PettingZoo) | experimental |
| `MA-DVRP-v0` | Multi-agent dynamic VRP with wait | experimental |
| `VRPSD-Outsource-v0` | Stochastic demands with outsourcing (VRP-SDO) | planned |
| `EVRP-NL-v0` | Electric VRP with nonlinear charging | planned |

**Status.** `stable`: reproduces published results. `experimental`: implemented and tested, not yet
replicated against a benchmark. `planned`: the config validates but cannot be run in this version.

Default instances come from generators (convenient sizes, not benchmark settings); pass
`instance=` or a `files` instance source for benchmark runs. Any environment can be modified with
overrides or extended by a new YAML file: {doc}`guides/build-your-own-variant`.
