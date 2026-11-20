from pathlib import Path

import pytest

from fleetgym.errors import ConfigError
from fleetgym.spec.yaml_io import dump_yaml, load_yaml, load_yaml_file


def test_exponent_without_dot_is_a_float():
    assert load_yaml("a: 1e-6\nb: -2E3\nc: 1.5e2\nd: 7\ne: '1e-6'") == {
        "a": 1e-6,
        "b": -2000.0,
        "c": 150.0,
        "d": 7,
        "e": "1e-6",
    }


def test_duplicate_keys_are_rejected_with_line():
    with pytest.raises(ConfigError, match=r"line 3.*duplicate key 'a'"):
        load_yaml("a: 1\nb: 2\na: 3\n", source="x.yaml")


def test_nested_duplicates_are_rejected():
    with pytest.raises(ConfigError, match="duplicate key 'k'"):
        load_yaml("top:\n  k: 1\n  k: 2\n")


def test_syntax_errors_name_the_source():
    with pytest.raises(ConfigError, match=r"^bad\.yaml: invalid YAML"):
        load_yaml("a: [1, 2", source="bad.yaml")


def test_unsafe_tags_are_rejected():
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_yaml("!!python/object/apply:os.system ['echo hi']")


def test_load_file(tmp_path: Path):
    (tmp_path / "ok.yaml").write_text("a: 1\n")
    (tmp_path / "empty.yaml").write_text("")
    (tmp_path / "list.yaml").write_text("- 1\n")
    assert load_yaml_file(tmp_path / "ok.yaml") == {"a": 1}
    assert load_yaml_file(tmp_path / "empty.yaml") == {}
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_yaml_file(tmp_path / "list.yaml")
    with pytest.raises(ConfigError, match="not found"):
        load_yaml_file(tmp_path / "missing.yaml")


def test_dump_keeps_order():
    assert dump_yaml({"b": 1, "a": [1, 2]}) == "b: 1\na:\n- 1\n- 2\n"
