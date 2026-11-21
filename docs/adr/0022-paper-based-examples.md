# 0022: Paper-based examples (M13)

Status: accepted (2026-11-20).

## Context
The user asked for one example per environment that reproduces the problem setting of a published
paper (not its method or results), read from the paper itself, plus a complete notebook guide to
building a variant.

## Decision
| Example | Environment | Paper (sections read) |
| --- | --- | --- |
| `09_vrp_vcsd` | `VRP-VCSD-v0` | Dastpak, Errico and Jabali, arXiv 2109.10200 (3.1, 6.1.1, B.1.1) |
| `10_vrpsd_goodson` | `VRPSD-Goodson-v0` | the same paper (6.2, B.1.2), after Goodson, Thomas and Ohlmann (2016) |
| `11_cvrp_kool` | `CVRP-v0` | Kool, van Hoof and Welling, arXiv 1803.08475 (5.1, C.1) |
| `12_vrptw_solomon` | `VRPTW-v0` | Solomon (1987), instances and best-known solutions from SINTEF |
| `13_vrpsd_florio` | `VRPSD-v0` | Florio, Gendreau, Hartl, Minner and Vidal, arXiv 2302.02538 (3.1, 6.1) |
| `14_ma_cvrp_vera` | `MA-CVRP-v0` | Vera and Abad, arXiv 1912.03341 (3.1, 4, Table 1) |

Each notebook opens with the paper, the problem, and a table mapping the paper's definitions to the
config, and lists the differences (for example: Kool et al. do not state the distance metric; Vera
and Abad's text and algorithm disagree on the demand range, and 1 to 9 is the one consistent with
their capacities; Florio et al. optimize restocking, `VRPSD-v0` uses detour-to-depot).
`05_custom_variant` became a step-by-step guide (YAML variants, four building blocks tested on their
own, verification, a custom policy, a paired comparison, packaging).

## Deferred
`DVRP-StochReq-v0` and `MA-DVRP-v0`: no openly readable paper with a reproducible setting was found
(Ulmer et al. and Bono et al. are not openly accessible; Zhang et al. 2023 uses a street network;
Joe and Lau 2020 use proprietary data). They wait for the PDFs the user will provide.

## Consequences
The examples will be rewritten as presets when problem specs and instances are separated
(proposal 0001, ADR 0023).
