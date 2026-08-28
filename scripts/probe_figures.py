"""Figure 1: the whole paper in one image.

Three panels, one per probe, sharing a single vertical axis: win rate against the fixed expert.
Sharing the axis is the point. Capacity and the oracle observation are measured on the learner,
the value of information is measured on a search agent, and putting all three on one scale is what
turns three experiments into one verdict.

    python scripts/make_figures.py

Reads data/probes.json (scripts/collect_probes.py). An arm that has not finished is drawn as an
explicit gap with a note, never as a missing bar that a reader would not notice.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from hidden_info_survey.style import apply_style  # noqa: E402 - one style for the whole repo

# categorical slots 1 and 2 of the reference palette, in fixed order: baseline, then treatment
BASELINE = "#2a78d6"
TREATMENT = "#eb6834"
INK = "#1a1a19"
MUTED = "#6b6a63"
RULE = "#d8d6cc"


def style():
    """The repo-wide look, plus the two rcParams only this figure needs."""
    apply_style()
    plt.rcParams.update({"axes.titlesize": 10.5, "axes.titlepad": 26})


def _subtitle(ax, text):
    ax.annotate(text, (0, 1.02), xycoords="axes fraction", fontsize=8.2, color=MUTED,
                ha="left", va="bottom")


def _verdict(ax, text, color):
    """One line per panel saying what the numbers licensed. The figure has to survive a skim."""
    ax.annotate(text, (0.5, -0.235), xycoords="axes fraction", fontsize=8.6, color=color,
                ha="center", va="top", fontweight="bold")


def _seed_cloud(ax, x, values, color, label=None):
    """Every seed as a dot, the mean as a wide tick. Four seeds is too few to hide in a box."""
    offsets = [(i - (len(values) - 1) / 2) * 0.055 for i in range(len(values))]
    ax.scatter(
        [x + o for o in offsets],
        [v * 100 for v in values],
        s=34,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label=label,
    )
    mean = sum(values) / len(values) * 100
    ax.plot([x - 0.16, x + 0.16], [mean, mean], color=color, linewidth=2.2, zorder=4)
    ax.annotate(
        f"{mean:.1f}",
        (x + 0.19, mean),
        fontsize=7.5,
        color=color,
        va="center",
        fontweight="bold",
    )


def panel_capacity(ax, block):
    arms = block["arms"]
    for i, arm in enumerate(arms):
        _seed_cloud(ax, i, arm["win_rates"], BASELINE if i == 0 else TREATMENT)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a["label"] for a in arms], fontsize=8)
    ax.set_xlim(-0.5, len(arms) - 0.5)
    lo, hi = block["band_lo"] * 100, block["band_hi"] * 100
    ax.axhspan(lo, hi, color=MUTED, alpha=0.08, zorder=0)
    ax.set_title("1  Capacity", loc="left")
    _subtitle(ax, "is the network the limit?")
    _verdict(ax, "no: one band", BASELINE)
    ax.annotate(
        f"one band, {lo:.0f} to {hi:.0f}",
        (0.5, hi + 1.2),
        fontsize=7.5,
        color=MUTED,
        ha="center",
    )


def panel_value_of_information(ax, block):
    for series, color, label in (
        (block["fair"], BASELINE, "graded fairly"),
        (block["oracle"], TREATMENT, "reads hidden cards"),
    ):
        ax.plot(
            [r["rollouts"] for r in series],
            [r["win_rate"] * 100 for r in series],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xticks([r["rollouts"] for r in block["fair"]])
    ax.set_xticklabels([str(r["rollouts"]) for r in block["fair"]])
    ax.set_xlabel("search rollouts per move")
    ax.set_title("2  Value of information", loc="left")
    _subtitle(ax, "what is the hidden state worth?")
    _verdict(ax, "an upper bound only", TREATMENT)
    ax.legend(loc="upper left")


def panel_oracle(ax, block, pending):
    arms = block["arms"] if block else []
    # While the oracle arm is unfinished its file still holds whatever the last run wrote, which
    # may be a smoke test. Drawing that point would put a plumbing check on the same axis as a
    # result, so an incomplete arm is shown as absent rather than as zero.
    drawn = arms[:1] if pending else arms
    for i, arm in enumerate(drawn):
        _seed_cloud(ax, i, arm["win_rates"], BASELINE if i == 0 else TREATMENT)
    labels = [a["label"] for a in arms] or ["Baseline", "Oracle"]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["Baseline\n(4 planes)", "Oracle obs.\n(5 planes)"][: len(labels)],
                       fontsize=8)
    ax.set_xlim(-0.5, max(1, len(labels) - 1) + 0.5)
    ax.set_title("3  Oracle observation", loc="left")
    _subtitle(ax, "information-bound or learning-bound?")
    if not pending:
        _verdict(ax, "no benefit detected", INK)
    if pending:
        ax.annotate(
            "runs in flight\n(this panel is the paper)",
            (0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=8.5,
            color=MUTED,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=RULE),
        )


def main():
    style()
    with open(os.path.join(HERE, "data", "probes.json")) as f:
        probes = json.load(f)

    pending_oracle = any("oracle_observation" in p for p in probes.get("pending", []))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)

    panel_capacity(axes[0], probes["capacity"])
    panel_value_of_information(axes[1], probes["value_of_information"])
    panel_oracle(axes[2], probes.get("oracle_observation"), pending_oracle)

    axes[0].set_ylabel("win rate against the fixed expert (%)")
    for ax in axes:
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 95)

    fig.suptitle(
        "Three probes, one axis: what a plateau against a fixed expert is made of",
        x=0.005,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = os.path.join(HERE, "figures", "fig1_three_probes")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png")
    plt.close(fig)
    print(f"wrote {out}.pdf and .png")
    if pending_oracle:
        print("note: the oracle arm is incomplete, panel 3 is drawn as a placeholder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
