"""`fleetgym validate`: validate configs without running anything."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from fleetgym.errors import ConfigError
from fleetgym.spec import REGISTRY, load_spec

__all__ = ["add_parser"]


def add_parser(subparsers: Any) -> None:
    """Register ``fleetgym validate``."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "validate", help="check configs against the schema and merge rules"
    )
    parser.add_argument("configs", nargs="*", help="environment IDs or YAML paths")
    parser.add_argument(
        "--all", action="store_true", help="validate every registered config, planned included"
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace) -> int:
    targets = list(args.configs)
    if args.all:
        targets += list(REGISTRY.ids(include_planned=True))
    if not targets:
        print("error: give at least one config, or --all", file=sys.stderr)
        return 2
    failed = 0
    for target in targets:
        try:
            spec = load_spec(target, overrides=args.overrides)
        except ConfigError as exc:
            failed += 1
            print(f"FAIL {target}\n{exc}", file=sys.stderr)
            continue
        status = spec.meta.status
        reserved = spec.reserved_features()
        note = f"  reserved: {', '.join(reserved)}" if reserved else ""
        print(f"OK   {target}  [{status}]  {spec.short_hash()}{note}")
    return 1 if failed else 0
