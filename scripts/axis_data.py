"""Axis-sweep results, with the one validity rule that every consumer must apply.

Tables and figures both read this sweep. When they each loaded it themselves they drifted: the
appendix rejected a game the figure still plotted. One loader now serves both.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(directory=None):
    """Return (good, bad): {game: [runs]} for games that pass and fail the ceiling check.

    An arm cannot score above its own ceiling, because the ceiling is a closed-form property of the
    game rather than an estimate. When one does, the arm is not measuring what it claims. Observed
    in liars_dice 2d3, where the per-deal placebo tag was nearly unique and so became a fingerprint
    of the deal, leaking what the placebo exists to withhold.

    The test is on the mean gap against its own standard error, not an absolute tolerance. A single
    seed lands a hair above the ceiling routinely, because the arm is evaluated by sampling episodes
    while the ceiling is exact; averaged over seeds that noise cancels. An absolute threshold
    rejected four sound games.
    """
    directory = directory or os.path.join(HERE, "data", "axis_games")
    rows = defaultdict(list)
    for path in glob.glob(os.path.join(directory, "*.json")):
        r = json.load(open(path))
        rows[r["game"]].append(r)
    good, bad = {}, {}
    for game, runs in rows.items():
        violation = False
        for arm in ("baseline", "oracle", "placebo"):
            gaps = [r["arms"][arm]["gap_to_ceiling"] for r in runs if arm in r["arms"]]
            if len(gaps) < 2:
                continue
            sem = st.stdev(gaps) / math.sqrt(len(gaps))
            if st.mean(gaps) < -2 * max(sem, 1e-9):
                violation = True
        (bad if violation else good)[game] = runs
    return good, bad


def nice(game):
    """Short label for a game string, for tables and axes."""
    if game.startswith("liars_dice"):
        sides = game.split("dice_sides=")[1].rstrip(")")
        dice = game.split("numdice=")[1][0]
        return f"Liar's dice {dice}d{sides}" + (" IR" if "_ir" in game else "")
    return {"kuhn_poker": "Kuhn poker",
            "leduc_poker": "Leduc poker",
            "dark_hex(board_size=2)": "Dark hex 2x2 (solved)",
            "leduc_poker(action_mapping=true)": "Leduc action-map",
            "leduc_poker(players=2,suit_isomorphism=true)": "Leduc suit-iso"}.get(game, game)


if __name__ == "__main__":
    good, bad = load()
    print(f"  usable  {len(good)} games, {sum(len(v) for v in good.values())} runs")
    for g in sorted(bad):
        print(f"  REJECTED {g}: an arm scored above its own exact ceiling")
