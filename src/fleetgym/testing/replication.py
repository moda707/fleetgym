"""Model-fidelity replication: replay published solutions through an environment (design §5.2).

A best-known solution is turned into an action sequence for the ``next_node`` interface (route
by route, each followed by a return to the depot) and replayed. The replay must be feasible (every
action valid under the masks), the independent validator must accept the trajectory, and the
environment's objective value must equal the published cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fleetgym.data.instance import Instance
from fleetgym.data.solution import Solution
from fleetgym.envs.env import FleetGymEnv
from fleetgym.errors import InfeasibleActionError
from fleetgym.spec import ProblemSpec
from fleetgym.testing.validator import ValidationReport, validate

__all__ = ["ReplayResult", "replay_solution"]


@dataclass
class ReplayResult:
    """Outcome of replaying one published solution.

    Attributes
    ----------
    feasible
        Every action was valid and the episode ended.
    cost
        The environment's value of ``component`` (for example total distance).
    published
        The published cost.
    validation
        The independent validator's report.
    errors
        Problems found during the replay.
    """

    instance: str
    feasible: bool
    cost: float | None
    published: float | None
    validation: ValidationReport | None
    errors: list[str] = field(default_factory=list)

    def matches(self, tolerance: float) -> bool:
        """True if feasible, valid, and within ``tolerance`` (absolute) of the published cost."""
        return (
            self.feasible
            and self.validation is not None
            and self.validation.ok
            and self.cost is not None
            and self.published is not None
            and abs(self.cost - self.published) <= tolerance
        )


def replay_solution(
    spec: ProblemSpec, instance: Instance, solution: Solution, *, component: str = "distance"
) -> ReplayResult:
    """Replay ``solution`` (routes of instance node indices) through ``spec`` on ``instance``.

    Parameters
    ----------
    component
        The cost component compared with the published cost.
    """
    env = FleetGymEnv(spec, instance=instance, debug=True)
    env.reset(seed=0)
    result = ReplayResult(instance.name, False, None, solution.cost, None)
    terminated = False
    try:
        for route in solution.routes:
            for node in (*route, 0):
                _, _, terminated, _, _ = env.step(node)
    except InfeasibleActionError as exc:
        result.errors.append(f"infeasible action: {exc}")
        return result
    if not terminated:
        result.errors.append("the episode did not end after the last route")
        return result
    result.feasible = True
    result.cost = env.simulator.state.total_costs.get(component, 0.0)
    result.validation = validate(env.event_log, spec, instance)
    return result
