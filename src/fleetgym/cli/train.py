"""`fleetgym train`: train an SB3 agent on an environment (extra ``sb3``).

Example::

    fleetgym train CVRP-v0 --algo maskable_ppo --policy gat --steps 100000 --seed 0 --out runs/cvrp

Writes ``<out>/model.zip`` and ``<out>/run.json`` (environment, spec hash, algorithm, policy,
hyperparameters, seed, steps, package versions).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fleetgym.spec import load_spec

__all__ = ["add_parser"]


def add_parser(subparsers: Any) -> None:
    """Register ``fleetgym train``."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "train", help="train an SB3 agent (needs fleetgym[sb3])"
    )
    parser.add_argument("env", help="environment ID or YAML config")
    parser.add_argument("--algo", choices=("maskable_ppo", "maskable_dqn"), default="maskable_ppo")
    parser.add_argument("--policy", choices=("gat", "attention", "mlp"), default="gat")
    parser.add_argument("--steps", type=int, default=100_000, help="training steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("runs/latest"), help="output directory")
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace) -> int:
    from fleetgym.agents.maskable_dqn import MaskableDQN
    from fleetgym.agents.sb3 import make_sb3_vec, policy_settings, sb3_defaults
    from fleetgym.eval.report_card import _versions  # pyright: ignore[reportPrivateUsage]

    spec = load_spec(args.env)
    env_id = spec.id or str(args.env)
    vec = make_sb3_vec(args.env, args.n_envs, seed=args.seed)
    policy, policy_kwargs = policy_settings(args.algo, args.policy, embed_dim=args.embed_dim)
    hyper = sb3_defaults(env_id, args.algo)
    if args.algo == "maskable_ppo":
        from sb3_contrib import MaskablePPO

        hyper["n_steps"] = min(hyper["n_steps"], max(8, args.steps // args.n_envs))
        hyper["batch_size"] = min(hyper["batch_size"], hyper["n_steps"] * args.n_envs)
        model: Any = MaskablePPO(
            policy, vec, seed=args.seed, policy_kwargs=policy_kwargs, verbose=0, **hyper
        )
    else:
        hyper["learning_starts"] = min(hyper["learning_starts"], args.steps // 2)
        model = MaskableDQN(policy, vec, seed=args.seed, policy_kwargs=policy_kwargs, **hyper)
    model.learn(args.steps)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / "model.zip")
    run = {
        "env": env_id,
        "spec_hash": spec.content_hash(),
        "algo": args.algo,
        "policy": args.policy,
        "embed_dim": args.embed_dim,
        "hyperparameters": hyper,
        "seed": args.seed,
        "steps": args.steps,
        "n_envs": args.n_envs,
        "versions": _versions(),
    }
    (args.out / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    print(f"saved {args.out / 'model.zip'}")
    return 0
