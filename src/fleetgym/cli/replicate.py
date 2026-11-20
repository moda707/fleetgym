"""`fleetgym replicate`: replay published solutions through a config (design §5.2).

The collection comes from the config's ``meta.instances`` (``{source: cvrplib, set: X}`` means
``cvrplib/X``). Files are downloaded on demand and verified against the shipped manifest.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from fleetgym.data import read_instance, read_solution
from fleetgym.data.download import collections, fetch, load_collection
from fleetgym.errors import ConfigError
from fleetgym.spec import load_spec
from fleetgym.testing.replication import replay_solution

__all__ = ["add_parser", "collection_for"]


def collection_for(env: str) -> str:
    """The replication collection of a config, from ``meta.instances``."""
    spec = load_spec(env)
    meta = spec.meta.instances or {}
    source, set_name = meta.get("source"), meta.get("set")
    name = f"{source}/{set_name}" if set_name else str(source)
    if name not in collections():
        raise ConfigError(
            f"no replication data for {env}: collection {name!r} has no manifest "
            f"(available: {', '.join(collections())})",
            source=spec.source,
            key_path="meta.instances",
        )
    return name


def add_parser(subparsers: Any) -> None:
    """Register ``fleetgym replicate``."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "replicate", help="replay published best-known solutions (downloads data)"
    )
    parser.add_argument("config", help="environment ID")
    parser.add_argument("--instances", nargs="*", help="instance names (default: all)")
    parser.add_argument("--limit", type=int, help="replay only the first N instances")
    parser.add_argument(
        "--tolerance",
        type=float,
        help="absolute cost tolerance (default: the collection's, from its manifest)",
    )
    parser.set_defaults(handler=_run)


def _violations(result: Any) -> str:
    validation = result.validation
    return "; ".join(str(v) for v in validation.violations[:2]) if validation is not None else ""


def _run(args: argparse.Namespace) -> int:
    collection = load_collection(collection_for(args.config))
    names = list(args.instances or collection.names())
    if args.limit is not None:
        names = names[: args.limit]
    spec = load_spec(args.config)
    tolerance = collection.tolerance if args.tolerance is None else args.tolerance
    failures = 0
    print(f"{'instance':<16} {'published':>12} {'environment':>12}  result")
    for name in names:
        instance = read_instance(fetch(collection.name, name), fmt=collection.format)
        solution = read_solution(fetch(collection.name, name, kind="solution"))
        result = replay_solution(spec, instance, solution)
        if result.published is None:  # e.g. Solomon: the cost is in the table, not the file
            result.published = collection.published_cost(name)
        ok = result.matches(tolerance)
        failures += not ok
        cost = "-" if result.cost is None else f"{result.cost:.6g}"
        detail = "" if ok else f"  {'; '.join(result.errors) or _violations(result)}"
        print(f"{name:<16} {result.published!s:>12} {cost:>12}  {'OK' if ok else 'FAIL'}{detail}")
    print(
        f"{len(names) - failures}/{len(names)} instances reproduce the published cost",
        file=sys.stderr,
    )
    return 1 if failures else 0
