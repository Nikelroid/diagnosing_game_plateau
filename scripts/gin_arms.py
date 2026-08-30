"""The Gin Rummy arm statistics, read from the run files.

Table 2 in the main text and the arm table in the appendix both report these numbers. They used to
be computed in two places from two sources: the appendix read the twelve-seed run files while the
summary table read a stale four-seed data/probes.json. The two disagreed in the submitted draft
(24.7 vs 25.5 against 26.79 vs 27.04), which a reviewer caught. One reader now serves both.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st

CURRICULUM = os.path.join(os.environ.get("ADVCOEV_ROOT", os.path.expanduser("~/Adversarial-CoEvolution")), "sweep", "curriculum")

# arm label -> the run-file prefixes that make up that arm
ARMS = {
    "baseline": ("ppoarch_mlp_s", "basewide_s"),
    "oracle":   ("oracleobs_s",),
    "placebo":  ("placebo_s",),
}


def _win_rates(prefixes):
    out = []
    for pre in prefixes:
        for path in sorted(glob.glob(os.path.join(CURRICULUM, pre + "*.json"))):
            vs_gold = json.load(open(path)).get("vs_gold") or {}
            if vs_gold.get("win_rate") is not None:
                out.append(100 * vs_gold["win_rate"])
    return out


def arms():
    """{arm: {n, mean, sd, ci_lo, ci_hi, win_rates}} in percentage points."""
    out = {}
    for name, prefixes in ARMS.items():
        w = _win_rates(prefixes)
        if not w:
            continue
        mean, n = st.mean(w), len(w)
        sd = st.stdev(w) if n > 1 else 0.0
        # t at 95% on n-1 df, for the small n we actually have
        tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 8: 2.365, 11: 2.201, 12: 2.179}.get(n - 1, 2.2)
        half = tcrit * sd / math.sqrt(n) if n > 1 else 0.0
        out[name] = {"n": n, "mean": mean, "sd": sd, "win_rates": w,
                     "ci_lo": mean - half, "ci_hi": mean + half}
    return out


def contrast(a, b, dp=2):
    """Difference of two arms as an independent (Welch) estimate, in percentage points.

    The arms are independent training runs, not paired: seed i of the oracle arm shares no
    trajectory with seed i of the baseline, so a paired interval would understate the spread.

    The point estimate is computed from the means AS PRINTED, at `dp` decimals. Computing it at
    full precision instead makes the printed difference fail to match the printed means: a reader
    subtracting 27.02 from 27.04 gets -0.02 while the table says -0.03. Both are defensible and
    one of them is checkable, so we print the checkable one. The interval is unaffected, being
    wider than this by two orders of magnitude.
    """
    A, B = arms()[a], arms()[b]
    diff = round(A["mean"], dp) - round(B["mean"], dp)
    se = math.sqrt(A["sd"] ** 2 / A["n"] + B["sd"] ** 2 / B["n"])
    df = (A["sd"] ** 2 / A["n"] + B["sd"] ** 2 / B["n"]) ** 2 / (
        (A["sd"] ** 2 / A["n"]) ** 2 / (A["n"] - 1) + (B["sd"] ** 2 / B["n"]) ** 2 / (B["n"] - 1))
    # The critical value must come from the Welch degrees of freedom actually computed above.
    # An earlier version paired Welch df with t(11), which names one procedure and runs another.
    from scipy import stats
    tcrit = float(stats.t.ppf(0.975, df))
    return {"diff": diff, "se": se, "df": df, "tcrit": tcrit,
            "lo": diff - tcrit * se, "hi": diff + tcrit * se}


if __name__ == "__main__":
    a = arms()
    for k, v in a.items():
        print(f"  {k:<9} n={v['n']:2d}  mean={v['mean']:6.2f}  sd={v['sd']:.2f}  "
              f"95% CI [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]")
    print()
    for x, y in [("oracle", "baseline"), ("placebo", "baseline"), ("oracle", "placebo")]:
        c = contrast(x, y)
        print(f"  {x} - {y:<9} {c['diff']:+.2f} pp  95% CI [{c['lo']:+.2f}, {c['hi']:+.2f}]  "
              f"(df={c['df']:.1f})")
