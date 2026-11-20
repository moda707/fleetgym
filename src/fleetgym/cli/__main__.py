"""Entry point of the ``fleetgym`` command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from fleetgym._version import __version__
from fleetgym.cli import schema, show, validate
from fleetgym.errors import FleetGymError

_COMMANDS = {
    "list": show.add_list_parser,
    "show": show.add_show_parser,
    "validate": validate.add_parser,
    "schema": schema.add_parser,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with one subcommand per CLI command.
    """
    parser = argparse.ArgumentParser(
        prog="fleetgym",
        description="FleetGym: RL environments for vehicle routing.",
    )
    parser.add_argument("--version", action="version", version=f"fleetgym {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for add in _COMMANDS.values():
        add(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Parameters
    ----------
    argv
        Arguments without the program name; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 when a FleetGym error is reported.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except FleetGymError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
