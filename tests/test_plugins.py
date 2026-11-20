from types import SimpleNamespace

import pytest

import fleetgym.plugins as plugins
from fleetgym.plugins import PluginError, Registry


def test_register_and_get_with_decorator_and_direct_call():
    reg: Registry[object] = Registry("thing")

    @reg.register("a")
    class A: ...

    b = reg.register("b", object())
    assert reg.get("a") is A
    assert reg.get("b") is b
    assert reg.names() == ("a", "b")
    assert dict(reg.items()) == {"a": A, "b": b}
    assert "a" in reg
    assert "z" not in reg
    assert "2 entries" in repr(reg)


def test_duplicate_names_are_rejected():
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)
    with pytest.raises(PluginError, match="already registered"):
        reg.register("a", 2)


@pytest.mark.parametrize("name", ["", None])
def test_invalid_names_are_rejected(name):
    reg: Registry[int] = Registry("thing")
    with pytest.raises(PluginError, match="non-empty"):
        reg.register(name, 1)  # pyright: ignore[reportArgumentType]


def test_unknown_name_lists_known_names():
    reg: Registry[int] = Registry("thing")
    reg.register("alpha", 1)
    with pytest.raises(PluginError, match=r"unknown thing 'beta'; known: alpha"):
        reg.get("beta")


def test_load_plugins_imports_entry_points_once(monkeypatch: pytest.MonkeyPatch):
    imported: list[str] = []
    eps = [SimpleNamespace(name="p", value="my_plugin.module:register")]
    monkeypatch.setattr(plugins, "_plugins_loaded", False)
    monkeypatch.setattr(plugins, "entry_points", lambda group: eps)
    monkeypatch.setattr(plugins, "import_module", lambda name: imported.append(name))
    plugins.load_plugins()
    plugins.load_plugins()
    assert imported == ["my_plugin.module"]


def test_failing_plugin_raises_with_its_name(monkeypatch: pytest.MonkeyPatch):
    def boom(name: str) -> None:
        raise ImportError("missing dependency")

    monkeypatch.setattr(plugins, "_plugins_loaded", False)
    monkeypatch.setattr(
        plugins, "entry_points", lambda group: [SimpleNamespace(name="bad", value="x")]
    )
    monkeypatch.setattr(plugins, "import_module", boom)
    with pytest.raises(PluginError, match=r"'bad'.*missing dependency"):
        plugins.load_plugins()
