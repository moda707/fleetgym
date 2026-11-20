"""JSON report cards (design §23): what was run, on what, and how it did."""

from __future__ import annotations

import dataclasses
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fleetgym._version import __version__
from fleetgym.eval.harness import Evaluation
from fleetgym.eval.stats import compare

__all__ = ["report_card", "write_report_card"]

_PACKAGES = ("numpy", "scipy", "gymnasium", "pettingzoo", "pydantic", "pyvrp", "torch")


def _versions() -> dict[str, str]:
    out = {"fleetgym": __version__, "python": sys.version.split()[0]}
    for name in _PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def report_card(
    evaluation: Evaluation,
    *,
    reference: Evaluation | None = None,
    bound: bool | None = None,
    training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A JSON-serializable report card.

    Parameters
    ----------
    reference
        Another policy on the same seeds; adds a paired comparison and the gap to it.
    bound
        For anticipative results: whether the cost is a lower bound (exact hindsight) or a
        perfect-information heuristic (plan Q29). Required for anticipative evaluations.
    training
        Training budget and settings, as given.

    Raises
    ------
    ValueError
        If the evaluation is anticipative and ``bound`` is not given, or the reference was run on
        other seeds.
    """
    if evaluation.anticipative and bound is None:
        raise ValueError("say whether an anticipative result is a bound (bound=True/False)")
    card: dict[str, Any] = {
        "env_id": evaluation.env_id,
        "spec_hash": evaluation.spec_hash,
        "policy": evaluation.policy,
        "anticipative": evaluation.anticipative,
        "seeds": evaluation.seeds,
        "summary": dataclasses.asdict(evaluation.summary),
        "costs": evaluation.costs,
        "versions": _versions(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "training": training or {},
    }
    if evaluation.anticipative:
        card["kind"] = "perfect-information bound" if bound else "perfect-information heuristic"
    if reference is not None:
        if reference.seeds != evaluation.seeds:
            raise ValueError("the reference was evaluated on other seeds")
        comparison = compare(evaluation.costs, reference.costs)
        ref_mean = reference.summary.mean_cost
        card["reference"] = {
            "policy": reference.policy,
            "mean_cost": ref_mean,
            "gap": (evaluation.summary.mean_cost - ref_mean) / abs(ref_mean) if ref_mean else None,
            "paired": dataclasses.asdict(comparison),
        }
    return card


def write_report_card(card: dict[str, Any], path: str | Path) -> Path:
    """Write ``card`` as indented JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
