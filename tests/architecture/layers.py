"""Static checker for FleetGym's import layering (architecture rules 1 and 11).

The checker parses every module with ``ast`` (nothing is imported) and reports:

* imports of a higher layer;
* imports within the same layer that are not explicitly allowed;
* imports by the validator of anything outside layers L0 to L2 and ``uncertainty.scenario``;
* imports of heavy optional dependencies (torch, SB3, PyG, matplotlib, PyVRP) from layers L0 to L5.

The rules mirror ``docs/IMPLEMENTATION_PLAN.md`` section 2.1. Change both together.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT_PACKAGE = "fleetgym"

#: Layer of each top-level subpackage or module of ``fleetgym``.
LAYERS: dict[str, int] = {
    "errors": 0,
    "plugins": 0,
    "_hashing": 0,
    "_typing": 0,
    "_version": 0,
    "eventlog": 0,
    "data": 1,
    "spec": 2,
    "configs": 2,
    "network": 3,
    "uncertainty": 3,
    "sim": 4,
    "constraints": 4,
    "ev": 4,
    "interfaces": 5,
    "envs": 5,
    "baselines": 6,
    "agents": 6,
    "nn": 6,
    "eval": 7,
    "viz": 7,
    "testing": 7,
    "cli": 7,
}

#: The root package ``fleetgym/__init__.py`` is the public facade and may import anything;
#: nothing inside the library may import the facade.
FACADE_LAYER = 99

#: Allowed imports between *different* top-level packages of the same layer:
#: (importing package, imported module prefix).
SAME_LAYER_ALLOWED: frozenset[tuple[str, str]] = frozenset(
    {
        ("sim", "constraints"),
        ("constraints", "sim.state"),
        ("constraints", "sim.events"),
        ("constraints", "sim.decisions"),
        ("constraints", "sim.info_state"),
        ("envs", "interfaces"),
        ("agents", "nn"),
        ("cli", "eval"),
        ("cli", "viz"),
        ("cli", "testing"),
        ("viz", "eval"),
        ("eval", "testing"),  # the harness validates every episode (ADR 0017)
    }
)

#: Layer L0 modules may import each other freely.
L0_FREE = True

VALIDATOR_MODULE = "testing.validator"
VALIDATOR_MAX_LAYER = 2
VALIDATOR_EXTRA_ALLOWED = ("uncertainty.scenario",)

HEAVY_DEPENDENCIES = (
    "torch",
    "stable_baselines3",
    "sb3_contrib",
    "torch_geometric",
    "matplotlib",
    "pyvrp",
)
HEAVY_FORBIDDEN_MAX_LAYER = 5


@dataclass(frozen=True)
class Violation:
    """One forbidden import."""

    module: str
    line: int
    imported: str
    reason: str

    def __str__(self) -> str:
        return f"{self.module}:{self.line} imports {self.imported}: {self.reason}"


def _module_name(path: Path, src_root: Path) -> str:
    rel = path.relative_to(src_root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(module: str, is_package: bool, level: int, target: str | None) -> str:
    parts = module.split(".")
    base = parts if is_package else parts[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([*base, target] if target else base)


def _imported_names(tree: ast.AST, module: str, is_package: bool) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _resolve_relative(module, is_package, node.level, node.module)
            else:
                base = node.module or ""
            found.append((node.lineno, base))
            # `from fleetgym import sim` imports the subpackage `fleetgym.sim`.
            if base == ROOT_PACKAGE or base.startswith(ROOT_PACKAGE + "."):
                found.extend((node.lineno, f"{base}.{alias.name}") for alias in node.names)
    return found


def _top(internal: str) -> str:
    """Top-level component below the root package (``"sim"`` for ``"sim.engine"``)."""
    return internal.split(".", 1)[0] if internal else ""


def _layer(internal: str) -> int:
    if internal == "":
        return FACADE_LAYER
    top = _top(internal)
    if top not in LAYERS:
        raise KeyError(f"no layer assigned to fleetgym.{top}; add it to LAYERS")
    return LAYERS[top]


def _is_known_submodule(internal: str) -> bool:
    return internal == "" or _top(internal) in LAYERS


def find_violations(src_root: Path) -> list[Violation]:
    """Return every layering violation under ``src_root/fleetgym``.

    Parameters
    ----------
    src_root
        Directory that contains the ``fleetgym`` package (the ``src`` folder).
    """
    violations: list[Violation] = []
    package_dir = src_root / ROOT_PACKAGE
    for path in sorted(package_dir.rglob("*.py")):
        module = _module_name(path, src_root)
        is_package = path.name == "__init__.py"
        internal_module = module.removeprefix(ROOT_PACKAGE).lstrip(".")
        src_layer = _layer(internal_module)
        src_top = _top(internal_module)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for line, name in _imported_names(tree, module, is_package):
            root = name.split(".", 1)[0]
            if root != ROOT_PACKAGE:
                if src_layer <= HEAVY_FORBIDDEN_MAX_LAYER and root in HEAVY_DEPENDENCIES:
                    violations.append(
                        Violation(module, line, name, "heavy optional dependency in layers L0-L5")
                    )
                continue

            target = name.removeprefix(ROOT_PACKAGE).lstrip(".")
            if not _is_known_submodule(target):
                # `from fleetgym.x import Name` also yields "x.Name"; only judge real packages.
                continue
            if target == "" and src_layer != FACADE_LAYER:
                violations.append(Violation(module, line, name, "imports the public facade"))
                continue
            if src_layer == FACADE_LAYER:
                continue

            tgt_layer = _layer(target)
            tgt_top = _top(target)

            if internal_module == VALIDATOR_MODULE or internal_module.startswith(
                VALIDATOR_MODULE + "."
            ):
                allowed = tgt_layer <= VALIDATOR_MAX_LAYER or any(
                    target == extra or target.startswith(extra + ".")
                    for extra in VALIDATOR_EXTRA_ALLOWED
                )
                if not allowed:
                    violations.append(
                        Violation(
                            module, line, name, "validator must stay independent of the simulator"
                        )
                    )
                continue

            if tgt_layer > src_layer:
                violations.append(
                    Violation(module, line, name, f"upward import (L{src_layer} -> L{tgt_layer})")
                )
            elif tgt_layer == src_layer and tgt_top != src_top:
                if src_layer == 0 and L0_FREE:
                    continue
                if not any(
                    src_top == allowed_src and (target == prefix or target.startswith(prefix + "."))
                    for allowed_src, prefix in SAME_LAYER_ALLOWED
                ):
                    violations.append(
                        Violation(
                            module, line, name, f"same-layer import not allowed (L{src_layer})"
                        )
                    )
    # `from fleetgym.sim import engine` yields both "sim" and "sim.engine"; report a line once.
    unique: dict[tuple[str, int, str], Violation] = {}
    for violation in violations:
        unique.setdefault((violation.module, violation.line, violation.reason), violation)
    return list(unique.values())
