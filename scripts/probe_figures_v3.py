"""Four more figures: the sensitivity argument, the two apparatuses, the mechanism, and capacity.

Same visual system as v2: tinted fills with a saturated edge of the same hue, one soft grid
behind the data, serif text, direct labels instead of legend boxes wherever it fits.
"""
from __future__ import annotations
import glob, json, math, os, statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import importlib.util
_v2 = importlib.util.spec_from_file_location(
    "v2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_figures_v2.py"))
v2 = importlib.util.module_from_spec(_v2); _v2.loader.exec_module(v2)
INK, RULE, MUTED = v2.INK, v2.RULE, v2.MUTED
TEAL, TEAL_L, GOLD, GOLD_L = v2.TEAL, v2.TEAL_L, v2.GOLD, v2.GOLD_L
SLATE, SLATE_L, PLUM, PLUM_L = v2.SLATE, v2.SLATE_L, v2.PLUM, v2.PLUM_L
HERE, save, nice = v2.HERE, v2.save, v2.nice   # axis_data reached via v2.axis_data
MAIN = "$HOME/Adversarial-CoEvolution/sweep/curriculum"


def load_axis():
    """Shared loader, so a game rejected from the tables is not still drawn in a figure.

    fig_mechanism plots the very quantity the rejection rule tests, so a rejected game here would
    contradict the eleven-of-eleven claim the figure is cited for.
    """
    good, _ = v2.axis_data.load()
    return good


def load_neural():
    rows = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(HERE, "data", "neural", "*.json")):
        r = json.load(open(p)); rows[r["game"]][r["arm"]].append(r["mean_return_vs_cfr"])
    return rows


def gin_arms():
    def arm(pat):
        out = []
        for p in sorted(glob.glob(os.path.join(MAIN, pat))):
            d = json.load(open(p)); vg = d.get("vs_gold") or {}
            if vg.get("win_rate") is not None:
                out.append(100 * vg["win_rate"])
        return out
    return (arm("ppoarch_mlp_s*.json") + arm("basewide_s*.json"),
            arm("oracleobs_s*.json"), arm("placebo_s*.json"))


def fig_sensitivity():
    """The argument in one picture, on a scale that lets the games be compared.

    Raw effects cannot share an axis here: poker is measured in return and Gin Rummy in win
    rate, so a bar twice as long need not mean twice the effect. Everything is therefore shown
    as a standardised effect size, the difference in units of the seed-to-seed spread, which is
    dimensionless and answers the question the paper actually asks: could this probe have seen
    an effect if one were there?
    """
    ax_rows, nn = load_axis(), load_neural()
    items = []

    def d_and_ci(a, b):
        na, nb = len(a), len(b)
        sa = st.stdev(a) if na > 1 else 0.0
        sb = st.stdev(b) if nb > 1 else 0.0
        pooled = math.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / max(na+nb-2, 1)) or 1e-9
        d = (st.mean(a) - st.mean(b)) / pooled
        se = math.sqrt((na+nb)/(na*nb) + d*d/(2*(na+nb-2))) if na+nb > 2 else 0.5
        return d, 1.96*se, st.mean(a) - st.mean(b)

    for g, rs in ax_rows.items():
        o = [r["arms"]["oracle"]["mean_return_vs_cfr"] for r in rs]
        b = [r["arms"]["baseline"]["mean_return_vs_cfr"] for r in rs]
        d, ci, raw = d_and_ci(o, b)
        items.append((nice(g), d, ci, raw, "tabular", rs[0]["hidden_bits"]))
    for g, arms in nn.items():
        if "oracle" in arms and "baseline" in arms:
            d, ci, raw = d_and_ci(arms["oracle"], arms["baseline"])
            items.append((nice(g) + "  (neural)", d, ci, raw, "neural", None))
    items.sort(key=lambda t: t[1])

    b, o, _ = gin_arms()
    gd, gci, graw = d_and_ci(o, b)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axvline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axvspan(-0.8, 0.8, color=PLUM_L, alpha=0.55, lw=0, zorder=0)
    ax.errorbar(gd, 0, xerr=gci, fmt="none", ecolor=PLUM, elinewidth=1.8, capsize=3.5,
                capthick=1.4, zorder=3)
    ax.plot(gd, 0, "o", color=PLUM, mec="white", mew=1.2, ms=9, zorder=4)
    # gin arms are already stored as percentages, so graw is in percentage points
    ax.text(gd, -0.45, f"{graw:+.2f} pp", color=PLUM, fontsize=8.6,
            fontweight="bold", va="top", ha="center")
    for i, (lab, d, ci, raw, kind, bits) in enumerate(items, start=1):
        c = TEAL if kind == "tabular" else GOLD
        ax.errorbar(d, i, xerr=ci, fmt="none", ecolor=c, elinewidth=1.3, capsize=2.5,
                    capthick=1.1, alpha=0.75, zorder=3)
        ax.plot(d, i, "o", color="white", mec=c, mew=1.6, ms=6, zorder=4)
        ax.text(d + ci + 0.5, i, f"{raw:+.2f}", color=c, fontsize=8.0, va="center")
    labels = ["Gin Rummy   the paper's game"] + [
        (f"{t[0]}   {t[5]:.1f} bits" if t[5] else t[0]) for t in items]
    ax.set_yticks(range(len(items) + 1)); ax.set_yticklabels(labels, fontsize=8.6, color=INK)
    ax.get_yticklabels()[0].set_color(PLUM); ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_ylim(-0.8, len(items) + 0.8)
    ax.set_xlabel("standardised effect of handing the learner the hidden state "
                  "(difference in units of seed spread)")
    ax.set_title("The probe fires wherever the information is there, and not on Gin Rummy")
    # symlog: the poker effects are two orders of magnitude larger than Gin Rummy's, and a
    # linear axis that shows one hides the other
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xticks([-1, 0, 1, 10, 100])
    ax.set_xticklabels(["-1", "0", "1", "10", "100"])
    ax.xaxis.grid(True); ax.tick_params(axis="y", length=0)
    ax.text(1.05, len(items) + 0.6, "shaded band: too small to tell from no effect",
            fontsize=8.0, color=MUTED, ha="left", va="center")
    for c, lab in ((TEAL, "tabular learner"), (GOLD, "neural learner")):
        ax.plot([], [], "o", color="white", mec=c, mew=1.6, ms=6, label=lab)
    ax.legend(loc="lower left", handlelength=1.0, borderaxespad=0.8,
              bbox_to_anchor=(0.02, 0.06))
    ax.text(0.99, -0.155, "numbers beside each point are the raw effect",
            transform=ax.transAxes, fontsize=7.8, color=MUTED, ha="right")
    fig.tight_layout(); save(fig, "fig_sensitivity")


def fig_apparatus():
    """Two learners, two mechanisms for delivering the information, the same answer."""
    ax_rows, nn = load_axis(), load_neural()
    shared = [g for g in nn if g in ax_rows]
    if not shared:
        print("  [skip] apparatus: no overlap"); return
    shared.sort(key=lambda g: ax_rows[g][0]["hidden_bits"])
    x = np.arange(len(shared)); w = 0.36
    tab = [st.mean([r["oracle_minus_baseline"] for r in ax_rows[g]]) for g in shared]
    tab_e = [st.pstdev([r["oracle_minus_baseline"] for r in ax_rows[g]]) for g in shared]
    neu = [st.mean(nn[g]["oracle"]) - st.mean(nn[g]["baseline"]) for g in shared]
    neu_e = [math.sqrt(st.pstdev(nn[g]["oracle"])**2 + st.pstdev(nn[g]["baseline"])**2) / 2
             for g in shared]

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    for off, vals, errs, c, cl, lab in ((-w/2, tab, tab_e, TEAL, TEAL_L, "tabular, extra key"),
                                        (+w/2, neu, neu_e, GOLD, GOLD_L, "neural, extra channel")):
        ax.bar(x + off, vals, w, color=cl, edgecolor=c, lw=1.2, label=lab, zorder=3)
        ax.errorbar(x + off, vals, yerr=errs, fmt="none", ecolor=c, elinewidth=1.2,
                    capsize=3, capthick=1.1, zorder=4)
        for xx, v in zip(x + off, vals):
            ax.text(xx, v + 0.04, f"{v:.2f}", ha="center", fontsize=8.4,
                    fontweight="bold", color=c)
    ax.set_xticks(x); ax.set_xticklabels([nice(g) for g in shared], color=INK, fontsize=8.8)
    ax.set_ylabel("gain from the hidden state (return)")
    ax.set_title("Different learner, different mechanism, same answer")
    ax.yaxis.grid(True); ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper left", handlelength=1.2)
    ax.set_ylim(0, max(max(tab), max(neu)) * 1.3)
    fig.tight_layout(); save(fig, "fig_apparatus")


def _spread(items, gap):
    """Push end-of-line labels apart just enough not to overprint, keeping their order."""
    out, last = [], None
    for y, payload in sorted(items, key=lambda t: t[0]):
        if last is not None and y - last < gap:
            y = last + gap
        out.append((y, payload)); last = y
    return out


def fig_mechanism():
    """Why capture can exceed 100%: the oracle also changes how hard the task is to learn."""
    rows = load_axis()
    if not rows: 
        print("  [skip] mechanism: no data"); return
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    lim, ends = 0, []
    for g, rs in rows.items():
        gb = st.mean([r["arms"]["baseline"]["gap_to_ceiling"] for r in rs])
        go = st.mean([r["arms"]["oracle"]["gap_to_ceiling"] for r in rs])
        cap = st.mean([r["capture"] for r in rs if r.get("capture") is not None])
        c = TEAL if cap > 1 else GOLD
        lim = max(lim, gb, go)
        ax.plot([0, 1], [gb, go], "-", color=c, lw=1.6, alpha=0.75, zorder=3)
        ax.plot([0, 1], [gb, go], "o", color="white", mec=c, mew=1.5, ms=6, zorder=4)
        ends.append((go, (f"{nice(g)}  {cap:.0%}", c)))
    # many liar's dice variants finish at almost the same height, so the labels are spread
    for y, (lab, c) in _spread(ends, gap=lim * 0.062):
        ax.text(1.04, y, lab, color=c, fontsize=8.0, va="center")
    ax.set_xlim(-0.08, 1.95); ax.set_ylim(-0.02, lim * 1.22)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["without the oracle", "with the oracle"], color=INK)
    ax.set_ylabel("how far the learner sits below its own ceiling")
    ax.set_title("The oracle also changes how hard the task is to learn")
    ax.yaxis.grid(True); ax.tick_params(axis="x", length=0)
    ax.text(0.02, lim * 1.08, "down = easier to learn with the oracle (capture above 100%)",
            fontsize=8.2, color=MUTED)
    fig.tight_layout(); save(fig, "fig_mechanism")


def fig_capacity():
    """What a genuinely capacity-bound plateau looks like."""
    rows = defaultdict(dict)
    for p in glob.glob(os.path.join(HERE, "data", "capacity", "*.json")):
        r = json.load(open(p)); rows[r["game"]].setdefault(r["buckets"], []).append(r)
    if not rows:
        print("  [skip] capacity: no data"); return
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    cap_ends = []
    palette = [TEAL, GOLD, SLATE, PLUM]
    for (g, by_b), c in zip(sorted(rows.items()), palette):
        ks = sorted([k for k in by_b if k], key=int)
        xs = ks + [max(ks) * 3]                       # "exact" placed to the right of the sweep
        ys = [st.mean([r["exploitability"] for r in by_b[k]]) for k in ks]
        if 0 in by_b:
            ys.append(st.mean([r["exploitability"] for r in by_b[0]]))
        else:
            xs = ks
        ax.plot(xs, ys, "-", color=c, lw=1.9, zorder=3)
        ax.plot(xs, ys, "o", color="white", mec=c, mew=1.5, ms=5, zorder=4)
        cap_ends.append((ys[-1], (nice(g), c, xs[-1])))
    span = max(y for y, _ in cap_ends) - min(y for y, _ in cap_ends) or 1.0
    for y, (lab, c, xe) in _spread(cap_ends, gap=span * 0.09):
        ax.text(xe * 1.15, y, lab, color=c, fontsize=8.2, fontweight="bold", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("situations the learner can tell apart (log; rightmost point is exact)")
    ax.set_ylabel("exploitability (lower is better)")
    ax.set_title("What a capacity-bound plateau looks like")
    ax.yaxis.grid(True); ax.set_xlim(1.6, 6000)
    fig.tight_layout(); save(fig, "fig_capacity")


if __name__ == "__main__":
    fig_sensitivity(); fig_apparatus(); fig_mechanism(); fig_capacity()
