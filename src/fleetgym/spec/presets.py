"""Paper presets: a published problem setting, its grid of instances, and where they come from.

A preset (ADR 0023) names a model (a spec, usually extending a registered family), cites the paper,
lists the paper's grid (for example density x capacity), and says where each grid point's instances
come from: a benchmark collection (downloaded and verified), or a folder of instance files the user
provides. FleetGym ships no generated instances; each paper's example notebook has a function that
writes them in the format the preset expects.

Preset files live in ``fleetgym/presets/<preset>.yaml``::

    preset: dastpak2022-vcsd
    name: VRP with a highly variable customer basis and stochastic demands
    reference: {key: dastpak2022vcsd, title: ..., arxiv: "2109.10200", sections: ["3.1"]}
    spec: {extends: SVRP-v1, fleet: {...}, ...}
    grid:
      density: [very_low, low, moderate, high, very_high]
      capacity: [25, 50, 75]
    default: {density: moderate, capacity: 50}
    instances: {folder: "{density}-q{capacity}"}      # relative to the folder the user passes
    # or: instances: {collection: solomon/100, name: "{instance}"}
    notes: [...]
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from fleetgym.errors import ConfigError
from fleetgym.spec.loader import load_spec
from fleetgym.spec.models import CollectionSource, FilesSource, ProblemSpec
from fleetgym.spec.yaml_io import load_yaml_file

__all__ = ["Preset", "load_preset", "preset_names"]

_KEYS = {"preset", "name", "reference", "spec", "grid", "default", "instances", "notes"}


@dataclass(frozen=True)
class Preset:
    """A published problem setting.

    Attributes
    ----------
    id
        The preset's name, for example ``dastpak2022-vcsd``.
    name
        A readable title.
    reference
        The paper (key, title, arXiv id or DOI, sections read).
    spec_data
        The model, as a spec mapping (it may ``extends`` a registered family).
    grid
        The paper's grid: dimension names and their values, in order.
    default
        The grid point used when a dimension is not chosen.
    instances
        Where a grid point's instances come from: ``{folder: <template>}`` (a folder of instance
        files or an instance set, relative to the root the user gives) or ``{collection:
        <name>, name: <template>, ...}`` (a benchmark instance; other keys as in the
        ``collection`` instance source). Templates use the grid dimensions, ``{density}``.
    notes
        Differences from the paper and assumptions.
    """

    id: str
    name: str
    reference: Mapping[str, Any]
    spec_data: Mapping[str, Any]
    grid: Mapping[str, tuple[Any, ...]]
    default: Mapping[str, Any]
    instances: Mapping[str, Any]
    notes: tuple[str, ...] = ()
    source: Path | None = field(default=None, compare=False)

    def point(self, **choice: Any) -> dict[str, Any]:
        """A full grid point: ``choice`` completed with the defaults, checked against the grid.

        Raises
        ------
        ConfigError
            If a dimension is unknown or a value is not on the grid.
        """
        unknown = set(choice) - set(self.grid)
        if unknown:
            raise ConfigError(
                f"preset {self.id!r} has no dimension(s) {sorted(unknown)}; "
                f"dimensions: {', '.join(self.grid) or 'none'}"
            )
        point = {**self.default, **choice}
        for name, values in self.grid.items():
            if name not in point:
                raise ConfigError(
                    f"preset {self.id!r}: choose a value for {name!r}: {list(values)}"
                )
            if point[name] not in values:
                raise ConfigError(
                    f"preset {self.id!r}: {name}={point[name]!r} is not on the paper's grid "
                    f"{list(values)}"
                )
        return point

    def points(self) -> list[dict[str, Any]]:
        """Every grid point, in grid order."""
        names = list(self.grid)
        return [
            dict(zip(names, combo, strict=True)) for combo in itertools.product(*self.grid.values())
        ]

    def spec(self, instances: str | Path | None = None, **choice: Any) -> ProblemSpec:
        """The model with the instance source of a grid point.

        Parameters
        ----------
        instances
            Root folder of the instances (for folder presets): the folder the paper's example
            wrote. Not used by collection presets.
        **choice
            Grid values, for example ``density="high"``.

        Raises
        ------
        ConfigError
            For an unknown grid value, or a folder preset without ``instances``.
        """
        point = self.point(**choice)
        source: dict[str, Any]
        if "collection" in self.instances:
            source = {"type": "collection", **_fill(dict(self.instances), point)}
            CollectionSource.model_validate(source)
        else:
            if instances is None:
                raise ConfigError(
                    f"preset {self.id!r} needs its instances: pass instances=<folder>. The paper's "
                    "example notebook has a function that writes them."
                )
            root = Path(instances).expanduser().resolve()
            folder = root / str(self.instances["folder"]).format(**point)
            if not folder.exists():
                raise ConfigError(
                    f"preset {self.id!r}: no instances for {point} at {folder}. The paper's "
                    "example notebook has a function that writes them."
                )
            source = {"type": "files", "paths": [str(folder)]}
            FilesSource.model_validate(source)
        data = {**self.spec_data, "instance_source": source}
        return load_spec(data)


def _fill(value: Any, point: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**point)
    if isinstance(value, Mapping):
        return {k: _fill(v, point) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_fill(v, point) for v in value]
    return value


def _preset_dir() -> Path:
    return Path(str(resources.files("fleetgym"))) / "presets"


def _read(path: Path) -> Preset:
    data = load_yaml_file(path)
    where = str(path)
    unknown = set(data) - _KEYS
    if unknown:
        raise ConfigError(f"unknown preset keys {sorted(unknown)}", source=where)
    for key in ("preset", "spec", "instances"):
        if key not in data:
            raise ConfigError(f"a preset needs `{key}`", source=where, key_path=key)
    if path.stem != data["preset"]:
        raise ConfigError(
            f"the file name must equal the preset name {data['preset']!r}", source=where
        )
    instances = data["instances"]
    if not isinstance(instances, Mapping) or not ({"folder", "collection"} & set(instances)):
        raise ConfigError(
            "`instances` needs `folder` or `collection`", source=where, key_path="instances"
        )
    grid = {str(k): tuple(v) for k, v in (data.get("grid") or {}).items()}
    return Preset(
        id=str(data["preset"]),
        name=str(data.get("name", data["preset"])),
        reference=dict(data.get("reference") or {}),
        spec_data=dict(data["spec"]),
        grid=grid,
        default=dict(data.get("default") or {}),
        instances=dict(instances),
        notes=tuple(data.get("notes") or ()),
        source=path,
    )


def preset_names() -> tuple[str, ...]:
    """Names of the shipped presets, sorted."""
    return tuple(sorted(p.stem for p in _preset_dir().glob("*.yaml")))


def load_preset(name: str) -> Preset:
    """A shipped preset by name.

    Raises
    ------
    ConfigError
        If there is no such preset, or its file is malformed.
    """
    path = _preset_dir() / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"unknown preset {name!r}; known: {', '.join(preset_names()) or 'none'}")
    return _read(path)
