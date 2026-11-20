"""`fleetgym schema`: export the JSON Schema of config files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fleetgym.spec import json_schema

__all__ = ["add_parser", "render_schema"]


def render_schema() -> str:
    """Return the JSON Schema as formatted text (the format of the shipped schema file)."""
    return json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n"


def add_parser(subparsers: Any) -> None:
    """Register ``fleetgym schema``."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "schema", help="print or write the JSON Schema of config files"
    )
    parser.add_argument("-o", "--output", type=Path, help="write to a file instead of stdout")
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace) -> int:
    text = render_schema()
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0
