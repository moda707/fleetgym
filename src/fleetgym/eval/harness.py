"""Run a policy on a fixed scenario set, validating every episode (design §23)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetgym.baselines.protocol import Episode, Policy, run_episode
from fleetgym.envs.env import FleetGymEnv
from fleetgym.envs.factory import make
from fleetgym.errors import ValidationFailure
from fleetgym.eval.metrics import Summary, summarize
from fleetgym.spec import ProblemSpec
from fleetgym.testing.validator import validate

__all__ = ["Evaluation", "evaluate"]


@dataclass
class Evaluation:
    """Episodes and summary of one policy on one scenario set."""

    env_id: str
    spec_hash: str
    policy: str
    anticipative: bool
    seeds: list[int]
    episodes: list[Episode] = field(default_factory=list)

    @property
    def summary(self) -> Summary:
        """Summary metrics."""
        return summarize(self.episodes)

    @property
    def costs(self) -> list[float]:
        """Episode costs, in seed order."""
        return [e.cost for e in self.episodes]


def evaluate(
    env: str | Path | ProblemSpec | Mapping[str, Any] | FleetGymEnv,
    policy: Policy | Callable[[], Policy],
    seeds: Sequence[int],
    *,
    validate_episodes: bool = True,
    **env_kwargs: Any,
) -> Evaluation:
    """Play ``policy`` on every seed of ``seeds``.

    Parameters
    ----------
    env
        Environment, ID, YAML path, spec or spec dict.
    policy
        A policy, or a factory called once.
    validate_episodes
        Re-check each episode with the independent validator (raises `ValidationFailure`).
    """
    environment = env if isinstance(env, FleetGymEnv) else make(env, **env_kwargs)
    if isinstance(policy, type) or not hasattr(policy, "act"):
        chosen: Policy = policy()  # pyright: ignore[reportCallIssue]
    else:
        chosen = policy  # pyright: ignore[reportAssignmentType]
    if not isinstance(chosen, Policy):
        raise TypeError(f"{chosen!r} does not implement the Policy protocol")
    spec = environment.problem_spec
    result = Evaluation(
        env_id=spec.id or "",
        spec_hash=spec.content_hash(),
        policy=chosen.name,
        anticipative=chosen.anticipative,
        seeds=list(seeds),
    )
    for seed in seeds:
        episode = run_episode(environment, chosen, seed=seed)
        if validate_episodes:
            report = validate(
                environment.event_log,
                spec,
                environment.base_instance,
                scenario=environment.scenario,
            )
            if not report.ok:
                raise ValidationFailure(
                    f"{chosen.name} on seed {seed}: invalid episode: {report.violations[:3]}"
                )
        result.episodes.append(episode)
    return result
