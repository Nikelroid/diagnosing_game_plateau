"""Generate the appendix tables from the result files.

The workshop allows appendices with no page limit and reviewers are not required to read them,
so the appendix carries what a reader needs to check the paper rather than anything the argument
depends on. Every number is read from disk.
"""
from __future__ import annotations
import glob, json, math, os, statistics as st, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)          # so hidden_info_survey imports when run from scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for gin_arms

import axis_data
MAIN = os.path.join(os.environ.get("ADVCOEV_ROOT", os.path.expanduser("~/Adversarial-CoEvolution")), "sweep", "curriculum")
OUT = os.path.join(HERE, "tables")


def load_axis():
    """Delegates to scripts/axis_data.py, which figures read too."""
    return axis_data.load()


def esc(x):
    return str(x).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


nice = axis_data.nice   # shared so a game is labelled the same in every table and figure


def table_axis():
    rows, _ = load_axis()
    L = [r"\begin{tabular}{l r r r r r r}", r"\toprule",
         r"Game & Bits & $n$ & Worth & Banked & Capture & Widening \\", r"\midrule"]
    # A game whose tree exceeds the enumeration cap has no exact bit count. Report that rather
    # than substituting an estimate: the whole point of the atlas is that provenance is visible.
    for g, rs in sorted(rows.items(), key=lambda kv: (kv[1][0].get("hidden_bits") is None,
                                                      kv[1][0].get("hidden_bits") or 0)):
        w = st.mean([r["information_worth"] for r in rs])
        got = st.mean([r["oracle_minus_baseline"] for r in rs])
        cap = [r["capture"] for r in rs if r.get("capture") is not None]
        wid = [r["widening_cost"] for r in rs if r.get("widening_cost") is not None]
        hb = rs[0].get("hidden_bits")
        bits = f"{hb:.2f}" if hb is not None else "not computed"
        L.append(f"{esc(nice(g))} & {bits} & {len(rs)} & {w:.3f} & {got:.3f} & "
                 f"{100*st.mean(cap):.0f}\\% & {st.mean(wid):+.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "app_axis.tex"), "w").write("\n".join(L) + "\n")
    return len(rows)


def table_capacity():
    rows = defaultdict(dict)
    for p in glob.glob(os.path.join(HERE, "data", "capacity", "*.json")):
        r = json.load(open(p)); rows[r["game"]].setdefault(r["buckets"], []).append(r)
    ks = sorted({k for v in rows.values() for k in v}, key=lambda k: k if k else 10**9)
    L = [r"\begin{tabular}{l" + " r" * len(ks) + "}", r"\toprule",
         "Game & " + " & ".join("exact" if k == 0 else str(k) for k in ks) + r" \\",
         r"\midrule"]
    for g, by in sorted(rows.items()):
        cells = [f"{st.mean([r['exploitability'] for r in by[k]]):.2f}" if k in by else "--"
                 for k in ks]
        L.append(f"{esc(nice(g))} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "app_capacity.tex"), "w").write("\n".join(L) + "\n")
    return len(rows)


def table_gin():
    """The three Gin Rummy arms, plus every pairwise contrast.

    Reads scripts/gin_arms.py, the same source Table 2 in the main text reads, so the two cannot
    drift apart again.
    """
    import gin_arms
    a = gin_arms.arms()
    labels = {"baseline": "Baseline, four planes",
              "oracle": "Oracle, fifth plane is the real hand",
              "placebo": "Placebo, fifth plane carries no information"}
    L = [r"\begin{tabular}{l r r r l}", r"\toprule",
         r"Arm & $n$ & Mean & SD & 95\% CI \\", r"\midrule"]
    for k in ("baseline", "oracle", "placebo"):
        v = a[k]
        L.append(f"{labels[k]} & {v['n']} & {v['mean']:.2f} & {v['sd']:.2f} & "
                 f"$[{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]$ \\\\")
    L += [r"\midrule", r"\multicolumn{5}{l}{\emph{Contrasts, percentage points}} \\"]
    for x, y in (("oracle", "baseline"), ("placebo", "baseline"), ("oracle", "placebo")):
        c = gin_arms.contrast(x, y)
        L.append(f"{x} minus {y} & & {c['diff']:+.2f} & & "
                 f"$[{c['lo']:+.2f}, {c['hi']:+.2f}]$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "app_gin.tex"), "w").write("\n".join(L) + "\n")
    return sum(a[k]["n"] for k in a)


def table_budget():
    rows = defaultdict(list)
    for p in glob.glob(os.path.join(HERE, "data", "axis_budget", "*", "*.json")):
        r = json.load(open(p)); rows[(r["game"], r["episodes"])].append(r)
    L = [r"\begin{tabular}{l r r r r}", r"\toprule",
         r"Game & Episodes & Worth & Capture & Gap with oracle \\", r"\midrule"]
    for (g, ep), rs in sorted(rows.items()):
        cap = [r["capture"] for r in rs if r.get("capture") is not None]
        L.append(f"{esc(nice(g))} & {ep:,} & "
                 f"{st.mean([r['information_worth'] for r in rs]):.3f} & "
                 f"{100*st.mean(cap):.0f}\\% & "
                 f"{st.mean([r['arms']['oracle']['gap_to_ceiling'] for r in rs]):.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "app_budget.tex"), "w").write("\n".join(L) + "\n")
    return len(rows)


def table_gaps():
    """Per-game distance below each arm's own ceiling.

    This is what backs the claim that the sign of the gap difference predicts the direction of
    capture's bias. Printed per game so a reader can check the eleven-out-of-eleven directly.
    """
    rows, _ = load_axis()
    L = [r"\begin{tabular}{l r r r r}", r"\toprule",
         r"Game & Capture & Baseline gap & Oracle gap & Difference \\",
         r"\midrule"]
    agree = 0
    for g, rs in sorted(rows.items(), key=lambda kv: st.mean([r["capture"] for r in kv[1]])):
        cap = 100 * st.mean([r["capture"] for r in rs])
        bg = st.mean([r["arms"]["baseline"]["gap_to_ceiling"] for r in rs])
        og = st.mean([r["arms"]["oracle"]["gap_to_ceiling"] for r in rs])
        ok = (bg > og) == (cap > 100)
        agree += ok
        diff = round(bg, 3) - round(og, 3)      # so the column subtracts as printed
        L.append(f"{esc(nice(g))} & {cap:.0f}\\% & {bg:.3f} & {og:.3f} & ${diff:+.3f}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "app_gaps.tex"), "w").write("\n".join(L) + "\n")
    return f"{agree}/{len(rows)}"


def table_atlas():
    """All 88 atlas rows with their provenance, so the mixed estimators can be audited.

    Figure 1 cannot show which values are exact, which are combinatorial upper bounds and which
    are right-censored lower bounds. A reviewer asked for the table that can, and this is it.
    """
    import csv
    with open(os.path.join(HERE, "tables", "survey_full.csv")) as fh:
        rows = list(csv.DictReader(fh))
    short = {"closed form": "closed", "exact": "exact", "resampled": "resampled",
             "not measured": "none"}
    fam = {"card": "2p card", "multi": "3-4p card", "board": "board", "dice": "dice",
           "comm": "signalling", "solo": "solo", "dial": "gin dial"}
    def sortkey(r):
        try:
            return (0, -float(r["bits"]))
        except (TypeError, ValueError):
            return (1, 0)
    # 88 rows cannot fit a float, and a tabular inside one silently overflows the page: the
    # caption disappeared entirely from the built PDF. longtable breaks across pages instead and
    # repeats the header, so it carries its own caption rather than sitting in a table float.
    head = (r"Game & Engine & Family & Bits & Provenance \\" "\n" r"\midrule")
    L = [r"\begin{longtable}{l l l r l}",
         r"\caption{The full atlas, all 88 configurations. \emph{closed} is exact arithmetic for "
         r"the deal and an upper bound on the information set. \emph{exact} enumerates the tree. "
         r"\emph{censored floor} is a lower bound that exhausted its sample budget. "
         r"\emph{none} means no method applies. Compare two games only when their provenance "
         r"matches.}\label{tab:app-atlas}\\",
         r"\toprule", head, r"\endfirsthead",
         r"\multicolumn{5}{l}{\emph{The full atlas, continued}}\\",
         r"\toprule", head, r"\endhead",
         r"\midrule", r"\multicolumn{5}{r}{\emph{continued on the next page}}\\",
         r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    for r in sorted(rows, key=sortkey):
        try:
            bits = f"{float(r['bits']):.2f}"
        except (TypeError, ValueError):
            bits = "n/a"
        note = short.get(r["bits_source"], r["bits_source"])
        if r["bits_source"] == "resampled":
            try:
                if float(r["resample_censored_frac"]) > 0.5:
                    note = "censored floor"
            except (TypeError, ValueError):
                pass
        L.append(f"{esc(r['label'])} & {esc(r['engine'])} & {fam.get(r['family'], r['family'])} "
                 f"& {bits} & {note} \\\\")
    L += [r"\end{longtable}"]
    open(os.path.join(OUT, "app_atlas.tex"), "w").write("\n".join(L) + "\n")
    return len(rows)


def table_corrections():
    """Recompute the naive closed form for every catalog entry that needed the multiset counter.

    Generated rather than typed, so the correction table cannot drift from the catalog it documents.
    """
    import re
    from hidden_info_survey.hidden_bits import multiset_deal_bits, hand_bits
    from hidden_info_survey import catalog as C

    src = open(C.__file__).read()
    rows = []
    for m in re.finditer(
            r'GameSpec\("[^"]+",\s*"([^"]+)",\s*"[^"]*",\s*'
            r'((?:[\d.]+\s*\*\s*)?multiset_deal_bits\(\[.*?\],\s*\d+\))', src, re.S):
        label, expr = m.group(1), " ".join(m.group(2).split())
        new = eval(expr, {"multiset_deal_bits": multiset_deal_bits})
        tc = eval(re.search(r"\[(.*?)\]", expr, re.S).group(0))
        k = int(re.search(r"\],\s*(\d+)\)", expr).group(1))
        mult = float(expr.split("*")[0]) if "*" in expr.split("multiset")[0] else 1.0
        deck = sum(c * n for c, n in tc)
        rows.append((label, deck, k, int(mult), mult * hand_bits(deck, k), new))
    L = [r"\begin{tabular}{l r r r r r}", r"\toprule",
         r"Game & Deck & Hand & Hidden hands & Distinguishable & Multiset \\", r"\midrule"]
    for lab, deck, k, mu, old, new in rows:
        L.append(f"{esc(lab)} & {deck} & {k} & {mu} & {old:.1f} & \\textbf{{{new:.1f}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "app_corrections.tex"), "w").write("\n".join(L) + "\n")
    return len(rows)


if __name__ == "__main__":
    good, bad = load_axis()
    if bad:
        print("  REJECTED (an arm scored above its own exact ceiling):")
        for g in bad:
            print(f"    {g}")
    print(f"  axis: {table_axis()} games | capacity: {table_capacity()} games | "
          f"gin: {table_gin()} runs | budget: {table_budget()} cells | "
          f"atlas: {table_atlas()} rows | corrections: {table_corrections()} entries | gaps: {table_gaps()} predicted")
