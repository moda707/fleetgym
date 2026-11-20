# 0019: Visualization

Status: accepted (2026-11-20).

## Decision
- Every view reads the event log only (and the episode instance for coordinates): the **map**
  (customers colored by status at a time `t`, routes per vehicle up to `t`, vehicle positions,
  legend) and the **vehicle Gantt** chart (travel, waiting and service per vehicle, restocks as
  ticks). Matrix-only instances have no map (`ConfigError`).
- Render modes: `ansi` (text, as before), `rgb_array` (map and Gantt as an `(H, W, 3)` `uint8` array,
  drawn on an off-screen Agg canvas whatever the active backend) and `human` (a window; nothing is
  shown under a non-interactive backend). The environment (layer 5) never imports the visualization
  layer (7): it binds the two figure modes by name at render time, like a plugin entry point, and
  lists them in `metadata["render_modes"]` only when matplotlib is installed, so Gymnasium's
  `check_env` does not try them on installs without the `viz` extra.
- `fleetgym render <env> --policy --seed --out` plays a baseline on one scenario and saves the figure
  (png, pdf or svg by extension).
- Tests use the Agg backend and never compare pixels (brittle across matplotlib versions): shapes,
  dtype, a non-blank image, statuses and segments, and that rendering does not change the trajectory
  (identical event-log digests).

## CI
Default test jobs measure coverage of the core only (`.github/coverage-core.toml`, the 90% target of
CLAUDE.md for layers L0 to L5); modules that need an extra are tested by the extras job. Without this,
the default jobs failed the coverage gate after M9 although every test passed.

## Not in this milestone
Request timeline, load traces, uncertainty overlay, cost breakdown over time, policy inspector and
comparison views, the HTML replay viewer (out of scope for v0.1) and video recording helpers.
