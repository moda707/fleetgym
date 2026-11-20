"""Training callbacks (design §19): evaluation on fixed scenario sets and cost logging."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from fleetgym.agents.sb3 import SB3Policy
from fleetgym.baselines.protocol import run_episode
from fleetgym.envs.env import FleetGymEnv

__all__ = ["CostComponentLogger", "ScenarioEval"]


class ScenarioEval(BaseCallback):
    """Every ``every`` steps, play the model on fixed seeds and log the mean cost.

    For validated, reported evaluations use `fleetgym.eval.evaluate` after training.

    Parameters
    ----------
    env
        The evaluation environment.
    seeds
        Scenario seeds (for example `fleetgym.eval.scenario_seeds(env_id, "validation", 32)`).
    every
        Steps between evaluations.
    """

    def __init__(self, env: FleetGymEnv, seeds: Sequence[int], every: int = 10_000) -> None:
        super().__init__()
        self.eval_env = env
        self.seeds = list(seeds)
        self.every = every
        self.history: list[tuple[int, float]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            policy = SB3Policy(self.model)
            costs = [run_episode(self.eval_env, policy, seed=s).cost for s in self.seeds]
            mean = float(np.mean(costs))
            self.history.append((self.num_timesteps, mean))
            self.logger.record("eval/mean_cost", mean)
        return True


class CostComponentLogger(BaseCallback):
    """Log the mean of each cost component per finished episode (``costs/<component>``)."""

    def __init__(self) -> None:
        super().__init__()
        self._totals: dict[int, dict[str, float]] = {}

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.zeros(len(infos), dtype=bool))
        for i, (info, done) in enumerate(zip(infos, dones, strict=False)):
            total = self._totals.setdefault(i, {})
            for component, amount in info.get("costs", {}).items():
                total[component] = total.get(component, 0.0) + float(amount)
            if done:
                for component, amount in total.items():
                    self.logger.record_mean(f"costs/{component}", amount)
                self._totals[i] = {}
        return True
