"""Entry point of the ``fleetgym`` command."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from fleetgym._version import __version__


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
    parser.add_subparsers(dest="command", metavar="<command>")
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
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return 0  # pragma: no cover - subcommands are added from M1 on


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
