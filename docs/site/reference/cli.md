# Command line

| Command | Does |
| --- | --- |
| `fleetgym list` | list registered environments (`--all` includes planned ones) |
| `fleetgym show <env>` | print the resolved spec as YAML |
| `fleetgym validate <config>...` | validate configs; errors name the file and key path |
| `fleetgym schema [-o file]` | export the JSON Schema of config files |
| `fleetgym replicate <env>` | replay published best-known solutions (downloads data) |
| `fleetgym train <env> --algo --policy --steps` | train an SB3 agent (`fleetgym[sb3]`) |
| `fleetgym eval <env> --model DIR \| --baseline NAME` | evaluate on fixed test scenarios |
| `fleetgym render <env> --policy --seed --out` | save the map and Gantt figure (`fleetgym[viz]`) |

`fleetgym <command> --help` lists every option.
