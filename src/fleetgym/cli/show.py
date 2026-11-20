"""`fleetgym list` and `fleetgym show`: browse registered configs and print resolved ones."""

from __future__ import annotations

import argparse
from typing import Any

from fleetgym.spec import REGISTRY, load_spec, resolve_raw
from fleetgym.spec.yaml_io import dump_yaml

__all__ = ["add_list_parser", "add_show_parser"]


def add_list_parser(subparsers: Any) -> None:
    """Register ``fleetgym list``."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "list", help="list registered environments"
    )
    parser.add_argument("--all", action="store_true", help="include planned configs")
    parser.set_defaults(handler=_run_list)


def add_show_parser(subparsers: Any) -> None:
    """Register ``fleetgym show``."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "show", help="print a config after `extends` and overrides"
    )
    parser.add_argument("config", help="environment ID or path to a YAML file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a value, for example fleet.types.0.count=8 (repeatable)",
    )
    parser.add_argument(
        "--defaults", action="store_true", help="validate and include every default value"
    )
    parser.set_defaults(handler=_run_show)


def _run_list(args: argparse.Namespace) -> int:
    entries = REGISTRY.entries(include_planned=args.all)
    width = max((len(e.id) for e in entries), default=0)
    for entry in entries:
        print(f"{entry.id:<{width}}  {entry.status:<12}  {entry.family}")
    return 0


def _run_show(args: argparse.Namespace) -> int:
    if args.defaults:
        spec = load_spec(args.config, overrides=args.overrides)
        print(f"# spec hash: {spec.content_hash()}")
        print(spec.to_yaml(), end="")
    else:
        print(dump_yaml(resolve_raw(args.config, overrides=args.overrides)), end="")
    return 0
