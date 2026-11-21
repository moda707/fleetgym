"""When each uncertain value becomes known to the agent (design §6, revelation model).

Supported revelations per source in this version:

========================  ===================================================================
source                    revelations
========================  ===================================================================
``demand``                ``a_priori``, ``on_request``, ``on_departure``, ``on_arrival``
``service_time``          ``a_priori``, ``on_request``, ``on_departure``, ``on_arrival``,
                          ``on_completion``
``travel_time``           ``on_arrival`` (per arc: known when the vehicle arrives)
========================  ===================================================================

``on_request`` equals ``a_priori`` for customers known at the start (static requests). Demand
revealed ``on_completion`` or ``never`` is rejected: the vehicle learns the demand while serving it.
``never`` for service times is rejected for the same reason (the vehicle's clock reveals it).
"""

from __future__ import annotations

from fleetgym.errors import ConfigError
from fleetgym.spec.models import ProblemSpec, Revelation

__all__ = ["ALLOWED_REVELATIONS", "AT_START", "check_revelations"]

ALLOWED_REVELATIONS: dict[str, frozenset[Revelation]] = {
    "demand": frozenset(
        {
            Revelation.A_PRIORI,
            Revelation.ON_REQUEST,
            Revelation.ON_DEPARTURE,
            Revelation.ON_ARRIVAL,
        }
    ),
    "service_time": frozenset(
        {
            Revelation.A_PRIORI,
            Revelation.ON_REQUEST,
            Revelation.ON_DEPARTURE,
            Revelation.ON_ARRIVAL,
            Revelation.ON_COMPLETION,
        }
    ),
    "travel_time": frozenset({Revelation.ON_ARRIVAL}),
}

#: Revelations that make a static customer's value known at reset.
AT_START = frozenset({Revelation.A_PRIORI, Revelation.ON_REQUEST})


def check_revelations(spec: ProblemSpec) -> None:
    """Raise `ConfigError` for a source or revelation this version does not simulate."""
    for name, uncertain in spec.uncertainty.items():
        allowed = ALLOWED_REVELATIONS.get(name)
        if allowed is None:
            raise ConfigError(
                f"uncertainty source {name!r} is not simulated in this version",
                source=spec.source,
                key_path=f"uncertainty.{name}",
            )
        if uncertain.revealed not in allowed:
            options = ", ".join(sorted(str(r) for r in allowed))
            raise ConfigError(
                f"{name} cannot be revealed {str(uncertain.revealed)!r}; use one of: {options}",
                source=spec.source,
                key_path=f"uncertainty.{name}.revealed",
            )
