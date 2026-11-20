from pathlib import Path

import pyarrow.parquet as pq
import pytest

from fleetgym.errors import FleetGymError
from fleetgym.eventlog import EVENT_KINDS, LOG_SCHEMA, EventLog


def test_append_validates_kinds_and_columns():
    log = EventLog({"env_id": "X-v0"})
    log.append("depart", time=0.0, vehicle=0, from_node=0, node=3, value=2.5)
    assert log.rows == [
        {
            "seq": 0,
            "kind": "depart",
            "time": 0.0,
            "vehicle": 0,
            "from_node": 0,
            "node": 3,
            "value": 2.5,
        }
    ]
    with pytest.raises(FleetGymError, match="unknown event kind"):
        log.append("teleport")
    with pytest.raises(FleetGymError, match="unknown event log column"):
        log.append("arrive", colour="red")
    assert log.metadata["fleetgym_log_version"] == "1"
    assert len(log) == 1
    assert [r["kind"] for r in log] == ["depart"]


def test_truncate_copy_and_digest():
    log = EventLog()
    for t in range(3):
        log.append("arrive", time=float(t), vehicle=0, node=t)
    copy = log.copy()
    log.truncate(1)
    assert len(log) == 1 and len(copy) == 3
    assert log.digest() != copy.digest()
    copy.truncate(1)
    assert log.digest() == copy.digest()


def test_parquet_round_trip_with_nulls(tmp_path: Path):
    log = EventLog({"spec_hash": "fg1:abc"})
    log.append("episode_start", time=0.0)
    log.append("cost", time=1.0, component="distance", amount=2.0)
    log.append("unserved", node=4)
    path = tmp_path / "log.parquet"
    log.write(path)
    table = pq.read_table(path)
    assert table.schema.equals(LOG_SCHEMA)
    again = EventLog.read(path)
    assert again.rows == log.rows
    assert again.metadata["spec_hash"] == "fg1:abc"
    assert len(log.of_kind("cost", "unserved")) == 2


def test_reading_an_incompatible_version_fails(tmp_path: Path):
    log = EventLog({"fleetgym_log_version": "0"})
    path = tmp_path / "old.parquet"
    log.write(path)
    with pytest.raises(FleetGymError, match="schema version"):
        EventLog.read(path)


def test_schema_documents_every_kind():
    from fleetgym.eventlog import schema

    assert schema.__doc__ is not None
    for kind in EVENT_KINDS:
        assert f"``{kind}``" in schema.__doc__
