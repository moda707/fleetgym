"""Event log schema and Parquet I/O (layer L0).

The log is the single record of an episode: the validator, rendering and evaluation read it.
This package holds only the schema and file I/O, so the validator can read logs without
importing the simulator. Implemented in M3.
"""
