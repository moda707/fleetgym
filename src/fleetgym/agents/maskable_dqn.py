"""DQN with action masking in selection and targets (design §19).

SB3's DQN has no action masking. `MaskableDQN` reads the mask from the observation
(``obs["action_mask"]``, present in every FleetGym observation):

* `MaskedQNetwork` sets the Q-values of masked actions to ``-inf``, so greedy selection never picks
  a masked action, and the target's ``max`` over next actions (computed by the target network on the
  next observation) ignores actions masked in the next state;
* epsilon-greedy exploration and warm-up actions are uniform among valid actions only;
* a next state with every action masked has a maximum of ``-inf``; it is replaced by 0 (such states
  are terminal in FleetGym, where at least one action stays valid otherwise).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MultiInputPolicy, QNetwork

__all__ = ["MaskableDQN", "MaskableDQNPolicy", "MaskedQNetwork"]


class MaskedQNetwork(QNetwork):
    """Q-network whose masked actions have Q-value ``-inf``."""

    def forward(self, obs: Any) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Q-values with masked actions at ``-inf``."""
        q = super().forward(obs)
        mask = obs["action_mask"].reshape(q.shape) > 0.5
        return q.masked_fill(~mask, float("-inf"))


class MaskableDQNPolicy(MultiInputPolicy):
    """DQN policy for dict observations with masked Q-networks."""

    def make_q_net(self) -> MaskedQNetwork:
        """A masked Q-network with its own features extractor."""
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQNetwork(**net_args).to(self.device)


class MaskableDQN(DQN):
    """DQN that never selects or bootstraps from masked actions.

    Use with ``policy=MaskableDQNPolicy`` (the default when ``policy="MaskableDQNPolicy"``).
    """

    policy_aliases = {"MaskableDQNPolicy": MaskableDQNPolicy}  # noqa: RUF012 - SB3 convention

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        observation: dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Epsilon-greedy over valid actions only.

        Uses NumPy's global generator, like SB3's DQN, so ``seed=`` makes training reproducible.
        """
        if not deterministic and np.random.rand() < self.exploration_rate:  # noqa: NPY002
            mask = np.asarray(observation["action_mask"]) > 0
            if mask.ndim == 1:
                return np.array(np.random.choice(np.flatnonzero(mask))), state  # noqa: NPY002
            return np.array([np.random.choice(np.flatnonzero(m)) for m in mask]), state  # noqa: NPY002
        return self.policy.predict(observation, state, episode_start, deterministic)

    def _sample_action(
        self, learning_starts: int, action_noise: Any = None, n_envs: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Warm-up actions are uniform over valid actions too (SB3 samples the whole space)."""
        if self.num_timesteps < learning_starts:
            assert isinstance(self._last_obs, dict)
            masks = np.asarray(self._last_obs["action_mask"]).reshape(n_envs, -1) > 0
            action = np.array([np.random.choice(np.flatnonzero(m)) for m in masks])  # noqa: NPY002
            return action, action
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """SB3's DQN update with masked targets (fully masked next states count 0)."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay.discounts if replay.discounts is not None else self.gamma
            with torch.no_grad():
                next_q, _ = self.q_net_target(replay.next_observations).max(dim=1)
                next_q = torch.where(torch.isfinite(next_q), next_q, torch.zeros_like(next_q))
                target = replay.rewards + (1 - replay.dones) * discounts * next_q.reshape(-1, 1)
            current = torch.gather(
                self.q_net(replay.observations), dim=1, index=replay.actions.long()
            )
            loss = F.smooth_l1_loss(current, target)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))
