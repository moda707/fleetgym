"""Compare policies on a fixed test set: confidence intervals, CVaR, paired tests, report card.

Run: python examples/04_compare_policies.py
"""

from pathlib import Path

from fleetgym.baselines import CheapestInsertion, NearestNeighbor
from fleetgym.eval import compare, evaluate, report_card, scenario_seeds, write_report_card

seeds = scenario_seeds("VRPSD-v0", "test", 16)  # fixed, never used for training
nearest = evaluate("VRPSD-v0", NearestNeighbor, seeds)  # every episode is validated
insertion = evaluate("VRPSD-v0", CheapestInsertion, seeds)

for result in (nearest, insertion):
    s = result.summary
    interval = f"[{s.ci_low:.1f}, {s.ci_high:.1f}]"
    print(f"{result.policy:>20}: {s.mean_cost:8.1f}  {interval}  CVaR90 {s.cvar90:.1f}")

paired = compare(insertion.costs, nearest.costs)
print(f"insertion - nearest: {paired.mean_difference:.1f} (t-test p = {paired.t_pvalue:.3f})")

card = report_card(insertion, reference=nearest)
path = write_report_card(card, Path("runs") / "example_report.json")
print("report card:", path)
