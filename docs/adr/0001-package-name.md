# 0001: Package name `fleetgym`

Status: accepted (2026-11-20)

## Context
The design document uses the working name "RouteGym" (`routegym`) and asks for its availability to be
checked. Checks on 2026-11-20:

- `routegym`: unregistered on PyPI, but `route-gym` exists (a 2020 shortest-route RL environment).
  PyPI rejects new project names that differ from an existing one only by separators, so `routegym`
  cannot be registered.
- `dynagym`: free on PyPI, but `SuReLI/dyna-gym` is an existing (inactive) RL package on GitHub, and the
  `dynagym` GitHub organization name is taken.
- `fleetgym`: free on PyPI (`fleetgym`, `fleet-gym`, `fleet_gym`) and as a GitHub organization name; no
  GitHub repository with that name.

## Decision
The user chose `fleetgym`. The distribution, import package, CLI command and Gymnasium namespace are all
`fleetgym` (`gymnasium.make("fleetgym/CVRP-v0")`). The prose name is FleetGym, and the conventional
import alias is `fg`. `docs/DESIGN.md` keeps "RouteGym" in its text; a header note maps the names.

## Alternatives
- `routegym` as the import name with a different distribution name: rejected, because a mismatch between
  `pip install` and `import` names confuses users.
- `dynagym`: rejected because of the GitHub name collision in the same field.

## Consequences
Names are first come, first served: the PyPI name and the GitHub organization should be reserved
before public announcements. The local folder is still named `dyna_gym`; renaming it is cosmetic.
