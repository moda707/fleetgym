---
name: Bug report
about: Something behaves differently from the documentation or the design
labels: bug
---

**What happened**

**What you expected**

**Reproduce** (environment ID or config, seed, and the shortest code that shows it)

```python
import fleetgym as fg
env = fg.make("CVRP-v0")
obs, info = env.reset(seed=0)
```

**Versions**: `python -c "import fleetgym; print(fleetgym.__version__)"`, Python, OS.
