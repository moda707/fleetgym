# Environment catalog

`fleetgym list` prints the registered model families; `fleetgym show <ID>` prints the resolved
YAML. A family says what exists; the instances give the numbers ({doc}`concepts/instances`).

| ID | Problem | Status |
| --- | --- | --- |
| `CVRP-v1` | Capacitated VRP; reproduces all CVRPLIB X best-known solutions exactly | stable |
| `VRPTW-v1` | VRP with hard time windows; reproduces Solomon best-known solutions | stable |
| `SVRP-v1` | Stochastic demands revealed on arrival, detour-to-depot recourse | experimental |
| `DVRP-v1` | Requests arriving during the day, accept/reject with cheapest insertion | experimental |
| `VRPSD-Outsource-v1` | Stochastic demands with outsourcing (VRP-SDO) | planned |
| `EVRP-NL-v1` | Electric VRP with nonlinear charging | planned |

Any family becomes multi-agent with `agents: {mode: multi}` and `fleet: {sizing: fixed}`
({doc}`concepts/multi-agent`).

**Status.** `stable`: reproduces published results. `experimental`: implemented and tested, not yet
replicated against a benchmark. `planned`: the config validates but cannot be run in this version.

## Presets

| Preset | Paper | Grid | Instances |
| --- | --- | --- | --- |
| `solomon1987-vrptw` | Solomon (1987) | 56 instances | downloaded (SINTEF) |
| `florio2023-vrpsd` | Florio et al. (2023), Section 6.1 | 4 CVRPLIB X instances | downloaded (CVRPLIB) |
| `kool2019-cvrp` | Kool et al. (2019), Appendix C.1 | n = 20, 50, 100 | example `11_cvrp_kool` |
| `dastpak2022-vcsd` | Dastpak, Errico and Jabali, Appendix B.1.1 | 5 densities x 3 capacities | example `09_vrp_vcsd` |
| `dastpak2022-goodson` | Goodson et al. (2016) via Dastpak et al., Appendix B.1.2 | 3 limits x 3 capacities x 3 variabilities | example `10_vrpsd_goodson` |
| `vera2019-hetfleet` | Vera and Abad (2019), Table 1 | VRP10, 20, 50, 80 | example `14_ma_cvrp_vera` |
| `ulmer2016-vrpssr` | Ulmer, Goodson, Mattfeld and Hennig (2016), Section 6.1 | 2 areas x 3 request rates x 3 location distributions | example `15_dvrp_ulmer` |

Any model can be modified with overrides or extended by a new YAML file:
{doc}`guides/build-your-own-variant`.
