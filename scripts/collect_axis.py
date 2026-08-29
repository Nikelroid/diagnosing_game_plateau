"""Gather the hidden-information-axis probes into one table.

    python scripts/collect_axis.py

Joins each game's probe result to the bits of hidden information the atlas measured for it,
so the oracle effect can be read against how much the game actually hides.
"""
from __future__ import annotations
import glob, json, os, statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BITS = {}          # atlas measurement, so the axis is not asserted from memory
for name in ("openspiel.json", "rlcard.json"):
    p = os.path.join(HERE, "data", name)
    if os.path.exists(p):
        for r in json.load(open(p))["rows"]:
            b = r.get("bits_exact") or r.get("bits_closed_form") or r.get("bits_resampled")
            if b is not None:
                BITS[r.get("game")] = (r["label"], b)


def load(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        out.setdefault((r["game"], r["episodes"]), []).append(r)
    return out


def report(title, rows):
    if not rows:
        print(f"\n{title}: nothing yet"); return
    print(f"\n{title}")
    print(f"  {'game':13s} {'bits':>5} {'episodes':>9} {'n':>2} "
          f"{'baseline':>9} {'oracle':>9} {'oracle-base':>12} {'expl(base)':>11}")
    for (game, eps), rs in sorted(rows.items(), key=lambda kv: (BITS.get(kv[0][0], ('', 0))[1], kv[0][1])):
        b = [r["arms"]["baseline"]["mean_return_vs_cfr"] for r in rs]
        o = [r["arms"]["oracle"]["mean_return_vs_cfr"] for r in rs]
        d = [x - y for x, y in zip(o, b)]
        e = [r["arms"]["baseline"].get("exploitability") for r in rs]
        e = [x for x in e if x is not None]
        bits = BITS.get(game, ("?", float("nan")))[1]
        sd = st.pstdev(d) if len(d) > 1 else 0.0
        print(f"  {game:13s} {bits:5.1f} {eps:9d} {len(rs):2d} "
              f"{st.mean(b):+9.4f} {st.mean(o):+9.4f} {st.mean(d):+8.4f}+/-{sd:4.3f} "
              f"{(st.mean(e) if e else float('nan')):11.4f}")


report("MAIN SWEEP (8 seeds per game)", load(os.path.join(HERE, "data", "axis", "*.json")))
report("BUDGET SWEEP (4 seeds per cell)", load(os.path.join(HERE, "data", "axis_budget", "*", "*.json")))

print("\nGin Rummy, from the study this paper builds on: 30.1 bits, oracle observation "
      "24.7% vs 25.5% baseline over 4 seeds, i.e. no detected gain.")
