"""Gather every number the paper uses into data/probes.json, once.

The runs live in the main repository, because that is where the training code and its result files
are. This script copies what the three probes need into this repository so figures, tables and the
paper can be rebuilt without it, and so the numbers in the paper have a single traceable source.

    python scripts/collect_probes.py [--main-repo ~/Adversarial-CoEvolution]

An arm that has not finished yet is reported as pending rather than silently omitted, so a figure
never quietly compares three seeds against one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from oracle_probes.probes import load_arm, load_ismcts_curve, summarise  # noqa: E402

DEFAULT_MAIN_REPO = os.environ.get(
    "ADVCOEV_ROOT", os.path.expanduser("~/Adversarial-CoEvolution")
)

# label -> glob. The MLP arm is the baseline the oracle arm is compared against: same recipe,
# same budget, same seeds, four planes instead of five.
CAPACITY_ARMS = [
    ("ppoarch_mlp_s*.json", "MLP"),
    ("ppoarch_attn_s*.json", "Self-attention"),
]
ORACLE_ARMS = [
    ("ppoarch_mlp_s*.json", "Baseline (4 planes)"),
    ("oracleobs_s*.json", "Oracle observation (5 planes)"),
]


def _check_budgets(arms, comparison, out):
    """Two arms trained for different numbers of steps are not a comparison.

    This is not hypothetical: a 20k-step smoke run writes to the same filename as the real 3M
    run, so without this check a plumbing test can end up in a figure as a result.
    """
    budgets = {arm.label: arm.steps for arm in arms}
    distinct = {steps for steps in budgets.values() if steps is not None}
    if len(distinct) > 1:
        out["pending"].append(
            f"{comparison}: step budgets disagree {budgets}; the shorter arm is not finished"
        )


def collect(results_dir):
    out = {"results_dir": results_dir, "pending": []}

    capacity = []
    for pattern, label in CAPACITY_ARMS:
        try:
            capacity.append(load_arm(results_dir, pattern, label))
        except (FileNotFoundError, ValueError) as exc:
            out["pending"].append(f"capacity/{label}: {exc}")
    if capacity:
        out["capacity"] = summarise(capacity)
        _check_budgets(capacity, "capacity", out)

    oracle = []
    for pattern, label in ORACLE_ARMS:
        try:
            oracle.append(load_arm(results_dir, pattern, label))
        except (FileNotFoundError, ValueError) as exc:
            out["pending"].append(f"oracle/{label}: {exc}")
    if oracle:
        out["oracle_observation"] = summarise(oracle)
        _check_budgets(oracle, "oracle_observation", out)

    fair = load_ismcts_curve(results_dir, oracle=False)
    cheat = load_ismcts_curve(results_dir, oracle=True)
    if fair and cheat:
        out["value_of_information"] = {
            "fair": fair,
            "oracle": cheat,
            "fair_range": [min(r["win_rate"] for r in fair), max(r["win_rate"] for r in fair)],
            "oracle_range": [min(r["win_rate"] for r in cheat), max(r["win_rate"] for r in cheat)],
        }
    else:
        out["pending"].append("value_of_information: ISMCTS result files not found")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-repo", default=DEFAULT_MAIN_REPO)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "probes.json"))
    args = ap.parse_args()

    results_dir = os.path.join(os.path.abspath(args.main_repo), "sweep", "curriculum")
    if not os.path.isdir(results_dir):
        print(f"results directory not found: {results_dir}", file=sys.stderr)
        return 2

    payload = collect(results_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"wrote {args.out}")
    for key in ("capacity", "oracle_observation"):
        block = payload.get(key)
        if not block:
            continue
        print(f"\n{key}: band {block['band_lo']:.3f} to {block['band_hi']:.3f}, "
              f"IQM {block['iqm']:.3f}")
        for arm in block["arms"]:
            rates = ", ".join(f"{r:.3f}" for r in arm["win_rates"])
            print(f"   {arm['label']:32s} n={arm['n']}  mean {arm['mean']:.3f}   [{rates}]")

    voi = payload.get("value_of_information")
    if voi:
        print("\nvalue of information (win rate vs the fixed expert)")
        for row_fair, row_oracle in zip(voi["fair"], voi["oracle"]):
            print(f"   {row_fair['rollouts']:4d} rollouts   fair {row_fair['win_rate']:.3f}   "
                  f"oracle {row_oracle['win_rate']:.3f}   "
                  f"({row_fair['seconds_per_move']:.2f} s/move fair)")

    if payload["pending"]:
        print("\nstill pending:")
        for item in payload["pending"]:
            print(f"   {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
