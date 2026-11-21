"""`fleetgym render`: play a baseline on one scenario and save the map and Gantt figure.

Example::

    fleetgym render DVRP-v1 --instances data/dvrp --policy nearest --seed 3 --out dvrp.png

Needs the ``viz`` extra (matplotlib). The file type follows the extension (png, pdf, svg).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fleetgym.baselines import run_episode
from fleetgym.cli.common import add_instances_argument, load_with_instances
from fleetgym.cli.eval import BASELINES
from fleetgym.envs.factory import make

__all__ = ["add_parser"]


def add_parser(subparsers: Any) -> None:
    """Register ``fleetgym render``."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "render", help="save the map and Gantt figure of one episode (needs fleetgym[viz])"
    )
    parser.add_argument("env", help="environment ID or YAML config")
    parser.add_argument("--policy", choices=sorted(BASELINES), default="nearest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("episode.png"))
    add_instances_argument(parser)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from fleetgym.viz import episode_figure

    env = make(load_with_instances(args.env, args.instances))
    episode = run_episode(env, BASELINES[args.policy](), seed=args.seed)
    figure = episode_figure(env)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    figure.savefig(args.out)
    print(f"{args.policy}: cost {episode.cost:.2f}; saved {args.out}")
    return 0
