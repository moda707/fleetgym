# Visualization

Every figure is drawn from the event log, so any episode can be drawn during or after the run
(`fleetgym[viz]`).

```{literalinclude} ../../../examples/08_render/render.py
:language: python
```

```bash
fleetgym render VRPTW-v1 --instances data/vrptw --policy insertion --seed 1 --out vrptw.png
```

Render modes: `ansi` (text), `rgb_array` (frames for videos) and `human` (a window).
