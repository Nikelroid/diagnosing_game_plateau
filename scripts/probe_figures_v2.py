"""Figures for the plateau decomposition.

Three panels, one job each:
  1. what the information is worth, and how much of it the learner banked
  2. the Gin Rummy verdict: three arms, twelve seeds, with intervals
  3. worth is fixed by the game while capture grows with budget

Styling follows the companion paper: tinted fills with a saturated edge of the same hue, one
soft grid behind the data, serif text so figures read as part of the page rather than beside it.
"""
from __future__ import annotations
import glob, json, math, os, sys, statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import axis_data
FIGS = os.path.join(HERE, "figures")

INK, RULE, MUTED = "#16242d", "#ccd4d8", "#77878f"
TEAL, TEAL_L = "#0b5b39", "#d6e6de"
GOLD, GOLD_L = "#a87a12", "#f6e6bf"
SLATE, SLATE_L = "#46606e", "#e6eaec"
PLUM, PLUM_L = "#6b3b6e", "#ecdff0"

_SERIF = next((c for c in ("Nimbus Roman", "Nimbus Roman No9 L", "Liberation Serif",
                           "FreeSerif", "DejaVu Serif")
               if any(c == f.name for f in fm.fontManager.ttflist)), "DejaVu Serif")
plt.rcParams.update({
    "font.family": "serif", "font.serif": [_SERIF], "font.size": 10,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.titlepad": 9,
    "axes.labelsize": 9.5, "axes.labelcolor": MUTED,
    "axes.edgecolor": RULE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "grid.color": RULE, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.8,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
SHORT = {"kuhn_poker": "Kuhn poker", "leduc_poker": "Leduc poker",
         "leduc_poker(action_mapping=true)": "Leduc (action map)",
         "leduc_poker(players=2,suit_isomorphism=true)": "Leduc (suit-iso)"}
def nice(g):
    if g in SHORT: return SHORT[g]
    if g.startswith("liars_dice"):
        n = g.split("dice_sides=")[1].rstrip(")")
        return f"Liar's dice, {n} sides" + (" (IR)" if "_ir" in g else "")
    return g


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=170)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


def fig_decomposition():
    rows, _ = axis_data.load()
    if not rows:
        print("  [skip] decomposition: no data"); return
    # A game whose tree exceeds the enumeration cap has no bit count and cannot be placed on this
    # axis; the table reports it as "not computed" instead.
    rows = {g: rs for g, rs in rows.items() if rs[0].get("hidden_bits") is not None}
    items = sorted(rows.items(), key=lambda kv: kv[1][0]["hidden_bits"])
    labels = [nice(g) for g, _ in items]
    worth = [st.mean([r["information_worth"] for r in rs]) for _, rs in items]
    got = [st.mean([r["oracle_minus_baseline"] for r in rs]) for _, rs in items]
    bits = [rs[0]["hidden_bits"] for _, rs in items]
    y = np.arange(len(items))

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.barh(y, worth, height=0.62, color=GOLD_L, edgecolor=GOLD, lw=1.1,
            label="what the information is worth", zorder=3)
    ax.barh(y, got, height=0.34, color=TEAL_L, edgecolor=TEAL, lw=1.1,
            label="what the learner banked", zorder=4)
    for i, (w, g) in enumerate(zip(worth, got)):
        ax.text(max(w, g) + 0.02, i, f"{100*g/w:.0f}%", va="center", ha="left",
                fontsize=8.6, fontweight="bold", color=TEAL if g / w <= 1 else PLUM)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}   {b:.1f} bits" for l, b in zip(labels, bits)], color=INK)
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.set_xlabel("expected return against the reference opponent")
    ax.set_title("Worth is set by the game; the share banked is not", pad=26)
    ax.xaxis.grid(True); ax.tick_params(axis="y", length=0)
    # legend above the plot as a strip: the bars run the full width, so any in-axes box collides
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              handlelength=1.3, columnspacing=1.6, borderpad=0.0)
    fig.tight_layout(); save(fig, "fig_decomposition")


def fig_gin():
    def arm(pat):
        out = []
        for p in sorted(glob.glob(os.path.join("$HOME/Adversarial-CoEvolution",
                                               "sweep", "curriculum", pat))):
            d = json.load(open(p)); vg = d.get("vs_gold") or {}
            if vg.get("win_rate") is not None:
                out.append(100 * vg["win_rate"])
        return out
    arms = [("baseline\nfour planes", arm("ppoarch_mlp_s*.json") + arm("basewide_s*.json"), SLATE, SLATE_L),
            ("oracle\nfifth plane, real hand", arm("oracleobs_s*.json"), GOLD, GOLD_L),
            ("placebo\nfifth plane, fake hand", arm("placebo_s*.json"), TEAL, TEAL_L)]
    arms = [a for a in arms if a[1]]
    if not arms:
        print("  [skip] gin: no data"); return
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for i, (lab, w, c, cl) in enumerate(arms):
        m, sd = st.mean(w), (st.stdev(w) if len(w) > 1 else 0.0)
        ci = 2.201 * sd / math.sqrt(len(w))
        ax.bar(i, m, 0.52, color=cl, edgecolor=c, lw=1.2, zorder=3)
        ax.errorbar(i, m, yerr=ci, color=c, lw=1.3, capsize=4, capthick=1.2, zorder=4)
        ax.scatter(np.full(len(w), i) + np.linspace(-.13, .13, len(w)), w,
                   s=13, color=c, alpha=0.45, zorder=5, linewidths=0)
        ax.text(i, m + ci + 0.55, f"{m:.2f}", ha="center", fontsize=9.2,
                fontweight="bold", color=c)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], color=INK, fontsize=8.8)
    ax.set_ylabel("win rate against the fixed expert (%)")
    ax.set_ylim(22, 31)
    ax.set_title("Gin Rummy: the hidden hand changes nothing, and neither does the widening")
    ax.yaxis.grid(True); ax.tick_params(axis="x", length=0)
    fig.tight_layout(); save(fig, "fig_gin_arms")


def fig_budget():
    rows = defaultdict(list)
    for p in glob.glob(os.path.join(HERE, "data", "axis_budget", "*", "*.json")):
        r = json.load(open(p)); rows[(r["game"], r["episodes"])].append(r)
    if not rows:
        print("  [skip] budget: no data"); return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.3})
    for game, c, cl in (("leduc_poker", TEAL, TEAL_L), ("kuhn_poker", GOLD, GOLD_L)):
        eps = sorted(e for (g, e) in rows if g == game)
        if not eps: continue
        worth = [st.mean([r["information_worth"] for r in rows[(game, e)]]) for e in eps]
        cap = [100 * st.mean([r["capture"] for r in rows[(game, e)] if r.get("capture")]) for e in eps]
        a1.plot(eps, worth, "-", color=c, lw=2.0, zorder=3)
        a1.plot(eps, worth, "o", color="white", mec=c, mew=1.5, ms=5, zorder=4)
        a1.text(eps[-1] * 1.15, worth[-1], nice(game), color=c, fontsize=8.8,
                fontweight="bold", va="center")
        a2.plot(eps, cap, "-", color=c, lw=2.0, zorder=3)
        a2.plot(eps, cap, "o", color="white", mec=c, mew=1.5, ms=5, zorder=4)
        a2.text(eps[-1] * 1.15, cap[-1], nice(game), color=c, fontsize=8.8,
                fontweight="bold", va="center")
    for ax, ttl, ylab in ((a1, "(a) Worth is fixed by the game", "information's worth (return)"),
                          (a2, "(b) The share banked grows with budget", "captured (%)")):
        ax.set_xscale("log"); ax.set_xlabel("training episodes")
        ax.set_ylabel(ylab); ax.set_title(ttl); ax.yaxis.grid(True)
        ax.set_xlim(2e4, 6e6)
    a1.set_ylim(0, 1.6); a2.set_ylim(40, 110)
    fig.subplots_adjust(wspace=0.34); save(fig, "fig_budget")


if __name__ == "__main__":
    fig_decomposition(); fig_gin(); fig_budget()
