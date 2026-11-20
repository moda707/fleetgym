"""Fleet size: enforced structurally by the simulator, not by a mask.

``fleet.sizing`` decides how many vehicles exist: ``fixed`` and ``limited`` fleets create
``count`` vehicles per type, and an ``unlimited`` fleet creates a new vehicle when the previous one
finishes (sequential epochs). A vehicle that does not exist cannot be dispatched, so no mask is
needed. ``fixed`` and ``limited`` behave the same in v0.1: vehicles may stay unused.
"""
