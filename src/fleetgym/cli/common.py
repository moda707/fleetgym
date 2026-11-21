"""Helpers shared by the commands that run episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

from fleetgym.spec import ProblemSpec, load_spec
from fleetgym.spec.models import FilesSource

__all__ = ["add_instances_argument", "load_with_instances"]


def add_instances_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--instances``: the problem instances to run on."""
    parser.add_argument(
        "--instances",
        type=Path,
        help="instance file, or folder of an instance set, to run on (replaces the config's "
        "instance_source; FleetGym ships no instances)",
    )


def load_with_instances(env: str, instances: Path | None) -> ProblemSpec:
    """The spec of ``env``, with ``instances`` as its instance source when given."""
    spec = load_spec(env)
    if instances is None:
        return spec
    source = FilesSource(paths=(str(instances.expanduser().resolve()),))
    return spec.model_copy(update={"instance_source": source})
