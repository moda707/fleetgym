"""Arrow schema of the event log and its metadata keys.

One row per event. Columns that do not apply to an event kind are null.

Event kinds and the columns they use
------------------------------------
``episode_start``
    ``time``.
``vehicle_activate``
    ``vehicle``, ``node`` (the depot), ``value`` (vehicle capacity), ``time``.
``decision``
    ``vehicle``, ``action`` (raw action), ``node`` (target node), ``epoch``.
``depart``
    ``vehicle``, ``from_node``, ``node`` (destination), ``time``, ``value`` (arc distance).
``arrive``
    ``vehicle``, ``node``, ``time``.
``service_start``
    ``vehicle``, ``node``, ``time``, ``quantity`` (demand served), ``load`` (load after service).
``service_end``
    ``vehicle``, ``node``, ``time``.
``route_end``
    ``vehicle``, ``node`` (the depot), ``time``, ``value`` (route duration).
``vehicle_retire``
    ``vehicle``, ``time``: the vehicle ends without leaving the depot.
``unserved``
    ``node``: a request left unserved at the end of the episode.
``reveal``
    ``node``, ``component`` (the uncertainty source), ``value`` (the realization), ``time``,
    optional ``vehicle``: the value becomes known to the agent.
``failure``
    ``vehicle``, ``node``, ``time``, ``quantity`` (demand still to deliver), ``load``,
    ``component`` (the resolution: ``return_to_depot`` or ``outsource``): a capacity failure and
    how it was resolved.
``restock``
    ``vehicle``, ``node`` (the depot), ``time``: the vehicle is emptied (load 0) without ending its
    route.
``outsource``
    ``vehicle``, ``node``, ``time``, ``quantity``, ``amount`` (its cost): demand handed to a third
    party (``component`` is ``request`` when a whole dynamic request is outsourced).
``request``
    ``node``, ``time``: a request is released (known to the agent).
``accept``
    ``node``, ``vehicle``, ``time``, ``value`` (insertion cost), ``action`` (plan position).
``reject``
    ``node`` (absent for an overflow request), ``time``, ``component`` (``decision`` or
    ``overflow``).
``cancel``
    ``node``, ``time``: the customer cancels its request.
``conflict``
    ``vehicle``, ``node``, ``time``, ``component`` (the conflict rule): in the multi-agent Parallel
    API, ``vehicle`` lost ``node`` to another vehicle (or the joint action was rejected).
``cost``
    ``component``, ``amount``, ``time``, optional ``vehicle`` and ``node``.
``epoch``
    ``epoch``, ``reward``, ``time``: the reward returned for the step that ended at this epoch.
``episode_end``
    ``time``.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = ["EVENT_KINDS", "LOG_SCHEMA", "LOG_SCHEMA_VERSION", "METADATA_KEYS"]

LOG_SCHEMA_VERSION = "2"

EVENT_KINDS = (
    "episode_start",
    "vehicle_activate",
    "decision",
    "depart",
    "arrive",
    "service_start",
    "service_end",
    "route_end",
    "vehicle_retire",
    "unserved",
    "reveal",
    "failure",
    "restock",
    "outsource",
    "request",
    "accept",
    "reject",
    "cancel",
    "conflict",
    "cost",
    "epoch",
    "episode_end",
)

LOG_SCHEMA = pa.schema(
    [
        pa.field("seq", pa.int64(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("time", pa.float64()),
        pa.field("vehicle", pa.int32()),
        pa.field("node", pa.int32()),
        pa.field("from_node", pa.int32()),
        pa.field("quantity", pa.float64()),
        pa.field("load", pa.float64()),
        pa.field("value", pa.float64()),
        pa.field("component", pa.string()),
        pa.field("amount", pa.float64()),
        pa.field("epoch", pa.int32()),
        pa.field("reward", pa.float64()),
        pa.field("action", pa.int64()),
    ]
)

#: Keys stored in the Parquet key-value metadata.
METADATA_KEYS = (
    "fleetgym_log_version",
    "fleetgym_version",
    "env_id",
    "spec_hash",
    "instance_hash",
    "instance_name",
    "episode_seed",
    "scenario_id",
)
