"""Event kinds, the event record and the event queue.

Events are ordered by ``(time, priority, seq)``. The priority breaks ties between events at the same
time in a fixed, documented order, and ``seq`` (insertion order) breaks the remaining ties, so event
processing is deterministic. All events with time ``<= t`` are processed before a decision epoch at
time ``t`` is presented.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum

__all__ = ["Event", "EventKind", "EventQueue"]


class EventKind(IntEnum):
    """Event kinds; the integer value is the tie-break priority (lower runs first)."""

    OVERFLOW_ARRIVAL = 4
    """A request beyond ``max_requests`` (``overflow: reject``); ``node`` is its overflow index."""
    REQUEST_ARRIVAL = 5
    CANCELLATION = 6
    VEHICLE_ARRIVAL = 10
    SERVICE_START = 15
    SERVICE_END = 20
    TIMER = 30
    """Wake-up of a waiting vehicle (planned routing: its latest time to head home)."""


@dataclass(frozen=True, order=True)
class Event:
    """A scheduled event.

    Attributes
    ----------
    time
        When the event happens.
    priority
        Tie-break priority (the kind's value).
    seq
        Insertion counter, the final tie-break.
    kind
        What happens.
    vehicle
        The vehicle concerned.
    node
        The node concerned.
    """

    time: float
    priority: int
    seq: int
    kind: EventKind = field(compare=False)
    vehicle: int = field(compare=False)
    node: int = field(compare=False)


@dataclass
class EventQueue:
    """A min-heap of events. Pure data, so a deep copy is an exact snapshot."""

    heap: list[Event] = field(default_factory=list)
    next_seq: int = 0

    def push(self, time: float, kind: EventKind, vehicle: int, node: int) -> Event:
        """Schedule an event and return it."""
        event = Event(time, int(kind), self.next_seq, kind, vehicle, node)
        self.next_seq += 1
        heapq.heappush(self.heap, event)
        return event

    def pop(self) -> Event:
        """Remove and return the earliest event."""
        return heapq.heappop(self.heap)

    def __len__(self) -> int:
        return len(self.heap)
