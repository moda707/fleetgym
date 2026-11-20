"""`fleetgym eval`: evaluate a trained model or a baseline on fixed test scenarios.

Examples::

    fleetgym eval CVRP-v0 --model runs/cvrp --episodes 32 --out runs/cvrp/report.json
    fleetgym eval CVRP-v0 --baseline nearest --episodes 32

Seeds come from `fleetgym.eval.scenario_seeds` (``--split``, default ``test``). Every episode is
validated. The report card is printed as JSON, and written to ``--out`` if given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fleetgym.baselines import CheapestInsertion, NearestNeighbor, RandomPolicy
from fleetgym.errors import ConfigError
from fleetgym.eval import evaluate, report_card, scenario_seeds, write_report_card
from fleetgym.spec import load_spec

__all__ = ["BASELINES", "add_parser"]

BASELINES: dict[str, Any] = {
    "random": RandomPolicy,
    "nearest": NearestNeighbor,
    "insertion": CheapestInsertion,
}


def add_parser(subparsers: Any) -> None:
    """Register ``fleetgym eval``."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "eval", help="evaluate a trained model or a baseline on fixed scenarios"
    )
    parser.add_argument("env", help="environment ID or YAML config")
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", type=Path, help="directory written by `fleetgym train`")
    who.add_argument("--baseline", choices=sorted(BASELINES))
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--reference", choices=sorted(BASELINES), help="baseline to compare with")
    parser.add_argument("--out", type=Path, help="write the report card here")
    parser.set_defaults(handler=_run)


def _model_policy(directory: Path) -> Any:
    from fleetgym.agents.maskable_dqn import MaskableDQN
    from fleetgym.agents.sb3 import SB3Policy

    run_file = directory / "run.json"
    if not run_file.exists():
        raise ConfigError(f"{directory} has no run.json; train with `fleetgym train`")
    run = json.loads(run_file.read_text())
    if run["algo"] == "maskable_ppo":
        from sb3_contrib import MaskablePPO

        model: Any = MaskablePPO.load(directory / "model.zip")
    else:
        model = MaskableDQN.load(directory / "model.zip")
    return SB3Policy(model, name=f"{run['algo']}-{run['policy']}"), run


def _run(args: argparse.Namespace) -> int:
    spec = load_spec(args.env)
    env_id = spec.id or str(args.env)
    seeds = scenario_seeds(env_id, args.split, args.episodes)
    training: dict[str, Any] = {}
    if args.model is not None:
        policy, run = _model_policy(args.model)
        if run["spec_hash"] != spec.content_hash():
            raise ConfigError(f"the model was trained on spec {run['spec_hash']}, not this one")
        training = run
    else:
        policy = BASELINES[args.baseline]()
    result = evaluate(args.env, policy, seeds)
    reference = evaluate(args.env, BASELINES[args.reference](), seeds) if args.reference else None
    card = report_card(result, reference=reference, training=training)
    if args.out is not None:
        write_report_card(card, args.out)
    print(json.dumps(card["summary"], indent=2))
    return 0
