from pathlib import Path
from types import ModuleType

import pytest

import fleetgym.spec.registry as registry_module
from fleetgym.errors import ConfigError
from fleetgym.spec import REGISTRY, ConfigRegistry
from tests.spec.helpers import BASE, write_yaml


def make_root(tmp_path: Path, *configs: tuple[str, dict]) -> Path:
    root = tmp_path / "root"
    for rel, data in configs:
        write_yaml(root / rel, data)
    return root


def test_scan_ids_status_family(tmp_path: Path):
    root = make_root(
        tmp_path,
        ("det/A-v0.yaml", {"id": "A-v0", **BASE}),
        ("dyn/B-v0.yaml", {"id": "B-v0", "meta": {"status": "planned"}, **BASE}),
        ("dyn/C-v1.yaml", {"id": "C-v1", "meta": {"status": "stable"}, **BASE}),
    )
    reg = ConfigRegistry([(root, "test")])
    assert reg.ids() == ("A-v0", "C-v1")
    assert reg.ids(include_planned=True) == ("A-v0", "B-v0", "C-v1")
    assert reg.status("A-v0") == "experimental"
    assert reg.status("B-v0") == "planned"
    assert reg.entry("C-v1").family == "dyn"
    assert reg.path("A-v0") == root / "det" / "A-v0.yaml"
    assert "A-v0" in reg and "Z-v0" not in reg
    assert [e.id for e in reg.entries(include_planned=False)] == ["A-v0", "C-v1"]


def test_unknown_id_suggests_close_matches(tmp_path: Path):
    reg = ConfigRegistry([(make_root(tmp_path, ("A/CVRP-v0.yaml", {"id": "CVRP-v0"})), "t")])
    with pytest.raises(ConfigError, match="did you mean CVRP-v0"):
        reg.entry("CVRP-v1")
    with pytest.raises(ConfigError, match=r"unknown environment ID 'XYZ'$"):
        reg.entry("XYZ")


@pytest.mark.parametrize(
    ("rel", "data", "match"),
    [
        ("A-v0.yaml", {"objective": {}}, "need an `id`"),
        ("A-v0.yaml", {"id": "B-v0"}, "expected B-v0.yaml"),
        ("A-v0.yaml", {"id": "A-v0", "meta": {"status": "beta"}}, "unknown status"),
    ],
)
def test_invalid_registry_files(tmp_path: Path, rel: str, data: dict, match: str):
    reg = ConfigRegistry([(make_root(tmp_path, (rel, data)), "t")])
    with pytest.raises(ConfigError, match=match):
        reg.ids()


def test_duplicate_ids_across_roots(tmp_path: Path):
    a = make_root(tmp_path / "1", ("A-v0.yaml", {"id": "A-v0"}))
    b = make_root(tmp_path / "2", ("x/A-v0.yaml", {"id": "A-v0"}))
    reg = ConfigRegistry([(a, "first"), (b, "second")])
    with pytest.raises(ConfigError, match=r"defined twice.*\(first\).*\(second\)"):
        reg.ids()


def test_refresh_rescans(tmp_path: Path):
    root = make_root(tmp_path, ("A-v0.yaml", {"id": "A-v0"}))
    reg = ConfigRegistry([(root, "t")])
    assert reg.ids() == ("A-v0",)
    write_yaml(root / "B-v0.yaml", {"id": "B-v0"})
    assert reg.ids() == ("A-v0",)
    reg.refresh()
    assert reg.ids() == ("A-v0", "B-v0")


def test_entry_points_add_config_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    callable_root = make_root(tmp_path / "c", ("X-v0.yaml", {"id": "X-v0"}))
    package_dir = tmp_path / "pkg" / "my_configs"
    write_yaml(package_dir / "Y-v0.yaml", {"id": "Y-v0"})
    (package_dir / "__init__.py").write_text("")
    module = ModuleType("my_configs")
    module.__path__ = [str(package_dir)]  # pyright: ignore[reportAttributeAccessIssue]
    module.__file__ = str(package_dir / "__init__.py")

    class EP:
        def __init__(self, name: str, target: object) -> None:
            self.name, self.value, self._target = name, f"{name}:x", target

        def load(self) -> object:
            return self._target

    monkeypatch.syspath_prepend(str(tmp_path / "pkg"))
    import my_configs  # pyright: ignore[reportMissingImports]

    eps = [EP("from_callable", lambda: callable_root), EP("from_package", my_configs)]
    monkeypatch.setattr(registry_module, "entry_points", lambda group: eps)
    reg = ConfigRegistry()
    ids = reg.ids(include_planned=True)
    assert {"X-v0", "Y-v0", "CVRP-v1"} <= set(ids)
    assert reg.entry("X-v0").origin == "entry point from_callable"


@pytest.mark.parametrize("target", [42, "broken"])
def test_bad_entry_points(monkeypatch: pytest.MonkeyPatch, target):
    class EP:
        name, value = "bad", "bad:x"

        def load(self) -> object:
            if target == "broken":
                raise ImportError("nope")
            return target

    monkeypatch.setattr(registry_module, "entry_points", lambda group: [EP()])
    with pytest.raises(ConfigError, match="'bad'"):
        ConfigRegistry().ids()


def test_package_registry_contains_the_shipped_configs():
    assert set(REGISTRY.ids()) == {"CVRP-v1", "VRPTW-v1", "SVRP-v1", "DVRP-v1"}
    assert set(REGISTRY.ids(include_planned=True)) >= {"VRPSD-Outsource-v1", "EVRP-NL-v1"}
