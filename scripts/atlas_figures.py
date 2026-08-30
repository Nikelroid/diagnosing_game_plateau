"""Build the paper figures from data/*.json. Nothing here computes a number.

    python scripts/make_figures.py

Three figures, one job each:
  1. how much is hidden, ranked          magnitude across many games -> horizontal bars
  2. hidden information against cost     two measures per game       -> scatter, log y
  3. the Gin Rummy dial                  one quantity moving         -> line

Figure 3 exists because it is the only comparison in the whole survey where the rules are held
fixed and a single knob moves, so a difference in outcome cannot be blamed on a difference in
game.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from hidden_info_survey.style import (  # noqa: E402
    ANCHOR,
    FAMILY_COLOR,
    FAMILY_LABEL,
    FAMILY_MARKER,
    INK,
    MUTED,
    apply_style,
    finish,
)

ANCHOR_LABEL = "Gin rummy (standard)"


def load_rows(data_dir):
    """Both engines, merged, with the best available hidden-bit estimate chosen per row."""
    rows = []
    for name in ("openspiel.json", "rlcard.json"):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print(f"  note: {name} missing, skipping")
            continue
        rows += json.load(open(path))["rows"]

    out = []
    for row in rows:
        if "error" in row:
            continue
        for key, source in (
            ("bits_exact", "exact"),
            ("bits_closed_form", "closed form"),
            ("bits_resampled", "resampled"),
        ):
            if row.get(key) is not None:
                row["bits"], row["bits_source"] = row[key], source
                break
        else:
            row["bits"], row["bits_source"] = None, "not measured"
        out.append(row)
    return out


def fig_hidden_bits(rows, out_dir):
    """Every game with a number, ranked. The reader's first question is 'who hides the most'."""
    data = sorted(
        [r for r in rows if r["bits"] is not None], key=lambda r: r["bits"], reverse=True
    )
    # Two columns rather than one. A single column of 45 bars is taller than a page, so LaTeX
    # floats it three pages away from the text that discusses it, which is worse than any
    # aesthetic concern the split introduces.
    half = (len(data) + 1) // 2
    chunks = [data[:half], data[half:]]
    height = max(3.6, 0.185 * half + 1.1)
    # Both panels share one x scale, so a bar in the right column is directly comparable with one
    # in the left. The wide gutter is not decoration: the right column's labels are long, and the
    # longest bar's value label needs somewhere to sit.
    fig, axes = plt.subplots(
        1, 2, figsize=(7.6, height), sharex=True, gridspec_kw={"wspace": 0.62}
    )
    xmax = max(r["bits"] for r in data) * 1.16

    for ax, chunk in zip(axes, chunks):
        for y, row in enumerate(chunk):
            if row["label"] == ANCHOR_LABEL:      # a band, not just bold text
                ax.axhspan(y - 0.46, y + 0.46, color=ANCHOR, alpha=0.10, zorder=1)
            color = FAMILY_COLOR[row["family"]]
            # a resampled value is a floor, not a measurement, so it is drawn hollow
            hollow = row["bits_source"] == "resampled"
            ax.barh(
                y,
                row["bits"],
                height=0.66,
                color="none" if hollow else color,
                edgecolor=color,
                linewidth=1.1 if hollow else 0.0,
                hatch="///" if hollow else None,
                zorder=3,
            )
            ax.text(
                row["bits"] + xmax * 0.015,
                y,
                f"{row['bits']:.1f}",
                va="center",
                ha="left",
                fontsize=6.6,
                color=MUTED,
            )

        ax.set_yticks(range(len(chunk)))
        ax.set_yticklabels([r["label"] for r in chunk], fontsize=6.8)
        for tick, row in zip(ax.get_yticklabels(), chunk):
            if row["label"] == ANCHOR_LABEL:
                tick.set_color(ANCHOR)
                tick.set_fontweight("bold")
        ax.set_ylim(len(chunk) - 0.5, -0.5)  # rank 1 at the top, no blank rows
        ax.set_xlim(0, xmax)
        ax.set_xlabel("support bits ($\\log_2$ of the histories still possible)")
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    families = [f for f in FAMILY_COLOR if any(r["family"] == f for r in data)]
    axes[1].legend(
        handles=[
            Line2D([], [], marker="s", linestyle="none", markersize=5,
                   color=FAMILY_COLOR[f], label=FAMILY_LABEL[f])
            for f in families
        ]
        + [Line2D([], [], marker="s", linestyle="none", markersize=5, markerfacecolor="none",
                  markeredgecolor=MUTED, label="hatched: censored floor")],
        loc="lower right",
        ncol=1,
        fontsize=6.8,
    )
    axes[0].set_title("Support bits: how much the player to move cannot see", loc="left", color=INK)
    finish(fig, os.path.join(out_dir, "fig1_hidden_bits"))


def fig_bits_vs_cost(rows, out_dir):
    """Hidden information against the learner's input size: the rung-selection chart."""
    # A handful of games declare a placeholder tensor rather than a real one; OpenSpiel's
    # cribbage reports an input of size 1. Those are not small observations, they are missing
    # ones, and plotting them invents a decade of empty axis.
    data = [r for r in rows if r["bits"] is not None and (r.get("input_size") or 0) >= 8]
    dropped = [
        r["label"] for r in rows
        if r["bits"] is not None and 0 < (r.get("input_size") or 0) < 8
    ]
    if dropped:
        print(f"  note: no usable observation tensor, omitted from figure 2: {', '.join(dropped)}")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    for family in FAMILY_COLOR:
        pts = [r for r in data if r["family"] == family]
        if not pts:
            continue
        ax.scatter(
            [r["bits"] for r in pts],
            [r["input_size"] for r in pts],
            s=42,
            marker=FAMILY_MARKER[family],
            facecolor=FAMILY_COLOR[family],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
            label=FAMILY_LABEL[family],
            zorder=3,
        )

    # Two labels only. Every extra one collided with a neighbour, and the cluster at 56.2 bits
    # is three games stacked on the same point, which a label cannot disambiguate anyway.
    anchor = next((r for r in data if r["label"] == ANCHOR_LABEL), None)
    top = max(data, key=lambda r: r["bits"])
    for row, dx, ha, color, weight in (
        (anchor, 8, "left", ANCHOR, "bold"),
        (top, -9, "right", MUTED, "normal"),
    ):
        if row is None:
            continue
        ax.annotate(
            row["label"],
            (row["bits"], row["input_size"]),
            textcoords="offset points",
            xytext=(dx, 4),
            ha=ha,
            fontsize=7.5,
            color=color,
            fontweight=weight,
        )

    ax.set_yscale("log")
    ax.set_xlabel("support bits ($\\log_2$ of the histories still possible)")
    ax.set_ylabel("learner input size (floats, log scale)")
    ax.set_xlim(-6, max(r["bits"] for r in data) * 1.08)
    ax.grid(True, which="major", zorder=0)
    ax.set_axisbelow(True)
    # the lower right is the one empty quadrant: cheap games that hide a lot do not exist
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("Information against the cost of learning it", loc="left", color=INK)
    finish(fig, os.path.join(out_dir, "fig2_bits_vs_cost"))


def fig_gin_dial(rows, out_dir):
    """The one controlled comparison: same rules, only the deck and hand size move."""
    dial = sorted(
        [r for r in rows if r["family"] == "dial" and r["bits"] is not None],
        key=lambda r: r["bits"],
    )
    if not dial:
        print("  note: no dial rows, skipping figure 3")
        return

    # x is the deck, not the hand: two configurations now share hand size 10 with different decks,
    # so hand size is no longer a unique key and its labels collide.
    def deck(row):
        if "deck " in row["label"]:
            return int(row["label"].split("deck ")[-1].split(",")[0])
        return 52  # the standard game

    dial = sorted(dial, key=deck)
    hands = [deck(r) for r in dial]
    bits = [r["bits"] for r in dial]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    # the filled area is the point of this figure: hidden information sweeps a factor of six
    # while every rule stays fixed
    ax.fill_between(hands, 0, bits, color=FAMILY_COLOR["dial"], alpha=0.10, zorder=1)
    ax.plot(hands, bits, color=FAMILY_COLOR["dial"], linewidth=2.4, marker="D",
            markersize=6, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    for hand, bit, row in zip(hands, bits, dial):
        emphasise = row["label"] == ANCHOR_LABEL
        ax.annotate(
            f"{bit:.1f}",
            (hand, bit),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=7.5,
            color=ANCHOR if emphasise else MUTED,
            fontweight="bold" if emphasise else "normal",
        )

    ax.set_xlabel("deck size (hand size grows with it)")
    ax.set_ylabel("support bits ($\\log_2$ of the histories still possible)")
    ax.set_ylim(0, max(bits) * 1.25)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("The dial: same rules, one knob", loc="left", color=INK)
    finish(fig, os.path.join(out_dir, "fig3_gin_dial"))


def main():
    apply_style()
    data_dir = os.path.join(HERE, "data")
    out_dir = os.path.join(HERE, "figures")
    os.makedirs(out_dir, exist_ok=True)

    rows = load_rows(data_dir)
    measured = [r for r in rows if r["bits"] is not None]
    print(f"{len(rows)} games, {len(measured)} with a hidden-information number")

    fig_hidden_bits(rows, out_dir)
    fig_bits_vs_cost(rows, out_dir)
    fig_gin_dial(rows, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
