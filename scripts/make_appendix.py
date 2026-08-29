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
MAIN = "$HOME/Adversarial-CoEvolution/sweep/curriculum"
OUT = os.path.join(HERE, "tables")


def esc(x):
    return str(x).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def nice(g):
    if g.startswith("liars_dice"):
        n = g.split("dice_sides=")[1].rstrip(")")
        d = g.split("numdice=")[1][0]
        return f"Liar's dice {d}d{n}" + (" IR" if "_ir" in g else "")
    return {"kuhn_poker": "Kuhn poker", "leduc_poker": "Leduc poker",
            "dark_hex(board_size=2)": "Dark hex 2x2 (solved)",
            "leduc_poker(action_mapping=true)": "Leduc action-map",
            "leduc_poker(players=2,suit_isomorphism=true)": "Leduc suit-iso"}.get(g, g)


def table_axis():
    rows = defaultdict(list)
    for p in glob.glob(os.path.join(HERE, "data", "axis_games", "*.json")):
        r = json.load(open(p)); rows[r["game"]].append(r)
    L = [r"\begin{tabular}{l r r r r r r}", r"\toprule",
         r"Game & Bits & $n$ & Worth & Banked & Capture & Widening \\", r"\midrule"]
    for g, rs in sorted(rows.items(), key=lambda kv: kv[1][0].get("hidden_bits") or 0):
        w = st.mean([r["information_worth"] for r in rs])
        got = st.mean([r["oracle_minus_baseline"] for r in rs])
        cap = [r["capture"] for r in rs if r.get("capture") is not None]
        wid = [r["widening_cost"] for r in rs if r.get("widening_cost") is not None]
        L.append(f"{esc(nice(g))} & {rs[0]['hidden_bits']:.2f} & {len(rs)} & {w:.3f} & {got:.3f} & "
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
    def arm(pat):
        out = []
        for p in sorted(glob.glob(os.path.join(MAIN, pat))):
            d = json.load(open(p)); vg = d.get("vs_gold") or {}
            if vg.get("win_rate") is not None:
                out.append(100 * vg["win_rate"])
        return out
    arms = [("Baseline, four planes", arm("ppoarch_mlp_s*.json") + arm("basewide_s*.json")),
            ("Oracle, fifth plane is the real hand", arm("oracleobs_s*.json")),
            ("Placebo, fifth plane carries no information", arm("placebo_s*.json"))]
    L = [r"\begin{tabular}{l r r r l}", r"\toprule",
         r"Arm & $n$ & Mean & SD & 95\% CI \\", r"\midrule"]
    for lab, w in arms:
        if not w:
            continue
        m, sd = st.mean(w), st.stdev(w)
        ci = 2.201 * sd / math.sqrt(len(w))
        L.append(f"{lab} & {len(w)} & {m:.2f} & {sd:.2f} & $[{m-ci:.2f}, {m+ci:.2f}]$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "app_gin.tex"), "w").write("\n".join(L) + "\n")
    return sum(len(w) for _, w in arms)


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
    rows = defaultdict(list)
    for p in glob.glob(os.path.join(HERE, "data", "axis_games", "*.json")):
        r = json.load(open(p)); rows[r["game"]].append(r)
    L = [r"\begin{tabular}{l r r r r c}", r"\toprule",
         r"Game & Capture & Baseline gap & Oracle gap & Difference & Predicted \\",
         r"\midrule"]
    agree = 0
    for g, rs in sorted(rows.items(), key=lambda kv: st.mean([r["capture"] for r in kv[1]])):
        cap = 100 * st.mean([r["capture"] for r in rs])
        bg = st.mean([r["arms"]["baseline"]["gap_to_ceiling"] for r in rs])
        og = st.mean([r["arms"]["oracle"]["gap_to_ceiling"] for r in rs])
        ok = (bg > og) == (cap > 100)
        agree += ok
        L.append(f"{esc(nice(g))} & {cap:.0f}\\% & {bg:.3f} & {og:.3f} & ${bg-og:+.3f}$ & "
                 f"{'yes' if ok else 'no'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "app_gaps.tex"), "w").write("\n".join(L) + "\n")
    return f"{agree}/{len(rows)}"


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
    print(f"  axis: {table_axis()} games | capacity: {table_capacity()} games | "
          f"gin: {table_gin()} runs | budget: {table_budget()} cells | "
          f"corrections: {table_corrections()} entries | gaps: {table_gaps()} predicted")
