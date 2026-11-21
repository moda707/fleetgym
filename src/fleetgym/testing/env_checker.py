"""Environment checks beyond Gymnasium's `check_env`.

`check_env` runs, for one config:

1. Gymnasium's ``check_env`` (with ``invalid_action="repair"``, because it steps random unmasked
   actions);
2. random masked rollouts whose event logs the independent validator must accept;
3. a determinism check (same seed, same event log);
4. a snapshot check (``clone`` mid-episode replays identically).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env as gymnasium_check_env

from fleetgym.envs.factory import make_any as make
from fleetgym.envs.gym_registration import NAMESPACE
from fleetgym.errors import ValidationFailure
from fleetgym.spec import ProblemSpec, load_spec
from fleetgym.testing.strategies import random_masked_rollout
from fleetgym.testing.validator import validate

__all__ = ["check_env"]


def check_env(env: str | Path | ProblemSpec, *, n_rollouts: int = 20, seed: int = 0) -> None:
    """Run every environment check on a config.

    Raises
    ------
    ValidationFailure
        If a rollout is invalid, episodes are not deterministic, or a clone diverges.
    AssertionError
        If Gymnasium's checker fails.
    """
    spec = load_spec(env)
    gym_id = f"{NAMESPACE}/{spec.id}"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if spec.id is not None and gym_id in gym.registry and load_spec(spec.id) == spec:
            registered = gym.make(gym_id, invalid_action="repair", render_mode="ansi")
            gymnasium_check_env(registered.unwrapped)
        else:
            # Without a Gymnasium registration there is no spec to re-make the env in other
            # render modes, so that one check is skipped.
            gymnasium_check_env(make(spec, invalid_action="repair"), skip_render_check=True)

    instance_env = make(spec)
    for k in range(n_rollouts):
        random_masked_rollout(instance_env, np.random.default_rng(seed + k), seed=seed + k)
        report = validate(
            instance_env.event_log,
            spec,
            instance_env.base_instance,
            scenario=instance_env.scenario,
        )
        if not report.ok:
            raise ValidationFailure(
                f"rollout with seed {seed + k} is invalid: {report.violations[:5]}"
            )

    digests = []
    for _ in range(2):
        twin = make(spec)
        random_masked_rollout(twin, np.random.default_rng(seed), seed=seed)
        digests.append(twin.event_log.digest())
    if digests[0] != digests[1]:
        raise ValidationFailure("the same seed produced different episodes")

    original = make(spec)
    original.reset(seed=seed)
    rng = np.random.default_rng(seed)
    original.step(int(rng.choice(np.flatnonzero(original.action_masks()))))
    clone = original.clone()
    for env_ in (original, clone):
        local = np.random.default_rng(seed + 1)
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env_.step(
                int(local.choice(np.flatnonzero(env_.action_masks())))
            )
    if original.event_log.digest() != clone.event_log.digest():
        raise ValidationFailure("a clone diverged from the original")
