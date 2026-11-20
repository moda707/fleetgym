# Multi-agent fleets

With `agents: {mode: multi}`, each vehicle is an agent of a [PettingZoo](https://pettingzoo.farama.org)
environment (`fleetgym.make_multi`):

- **AEC**: the vehicle with a decision acts, one at a time.
- **Parallel**: every vehicle that must decide now acts at once; busy vehicles send the no-op (the last
  action). When two vehicles choose the same customer, `agents.conflict` decides: `priority`,
  `distance`, `lottery` or `reject` (with a penalty).
- **Rewards**: `global`, `individual` (each vehicle's own costs plus a share of the rest) or `mixed`.
- `env.state()` gives the global state for a centralized critic.

```{literalinclude} ../../../examples/07_multi_agent.py
:language: python
```
