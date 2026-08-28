"""The three probes, read off result files.

Each probe answers one question about a plateau, and only the three together give a verdict:

  capacity          Is the network the limit?
                    Compare encoders trained on the identical recipe. If a plain MLP and a
                    self-attention encoder land in the same band, capacity is not what is missing.

  value of          How much is the hidden information worth at most?
  information       Compare a determinized search graded fairly against the same search allowed to
                    read the hidden cards. The gap is an upper bound on the value of that
                    information, and it is confounded upward by the removal of strategy fusion,
                    so it bounds rather than measures.

  oracle            Is the learner information-bound or learning-bound?
  observation       Retrain the identical recipe with the hidden state in the observation. If the
                    oracle learner does not improve, the plateau was never about information.

Nothing here computes a win rate. The runs produce them; this module only reads, groups and
summarises, so a number in the paper can always be traced back to a file.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass

__all__ = ["Arm", "load_arm", "load_ismcts_curve", "iqm", "summarise"]


@dataclass
class Arm:
    """One experimental arm: several seeds of the same recipe, graded against the fixed expert."""

    name: str
    label: str
    seeds: list[int]
    win_rates: list[float]  # best_vs_gold, one per seed
    steps: int | None = None

    @property
    def mean(self) -> float:
        return sum(self.win_rates) / len(self.win_rates)

    @property
    def spread(self) -> tuple[float, float]:
        return min(self.win_rates), max(self.win_rates)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(mean=round(self.mean, 4), lo=self.spread[0], hi=self.spread[1], n=len(self.seeds))
        return d


def iqm(values: list[float]) -> float:
    """Interquartile mean: the middle half, averaged. Less seed-sensitive than a plain mean."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) < 4:
        return sum(ordered) / len(ordered)
    cut = len(ordered) // 4
    middle = ordered[cut : len(ordered) - cut]
    return sum(middle) / len(middle)


def load_arm(results_dir: str, pattern: str, label: str, metric: str = "best_vs_gold") -> Arm:
    """Collect every seed of one arm, e.g. pattern 'ppoarch_mlp_s*.json'.

    Raises if the arm is empty, because a silently missing arm is how a figure ends up
    comparing three seeds against one.
    """
    paths = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"no result files match {pattern} in {results_dir}")

    seeds, rates, steps = [], [], None
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        if data.get(metric) is None:
            continue
        seeds.append(data.get("seed", len(seeds)))
        rates.append(float(data[metric]))
        steps = data.get("steps", steps)

    if not rates:
        raise ValueError(f"{pattern} matched {len(paths)} files but none carry '{metric}'")
    name = pattern.replace("_s*.json", "").replace("*.json", "")
    return Arm(name=name, label=label, seeds=seeds, win_rates=rates, steps=steps)


def load_ismcts_curve(results_dir: str, oracle: bool) -> list[dict]:
    """Win rate against the expert as a function of the search budget.

    `oracle=False` reads the determinized runs, which sample worlds consistent with what the
    searcher can see. `oracle=True` reads the variant that is handed the true hidden cards.
    """
    prefix = "ismcts_vs_gold_r" if oracle else "ismcts_det_vs_gold_r"
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"{prefix}*.json"))):
        with open(path) as f:
            data = json.load(f)
        result = data.get("vs_gold") or {}
        if "win_rate" not in result:
            continue
        games = result.get("n") or 1
        out.append(
            {
                "rollouts": data["rollouts"],
                "win_rate": result["win_rate"],
                "games": games,
                # eval_seconds covers both seats over the whole evaluation; mean_len is plies
                "seconds_per_move": data["eval_seconds"] / games / max(1.0, result.get("mean_len", 30)),
            }
        )
    return sorted(out, key=lambda r: r["rollouts"])


def summarise(arms: list[Arm]) -> dict:
    """The band every arm falls in, which is what the capacity probe actually reports."""
    everything = [rate for arm in arms for rate in arm.win_rates]
    return {
        "arms": [arm.as_dict() for arm in arms],
        "band_lo": min(everything),
        "band_hi": max(everything),
        "iqm": round(iqm(everything), 4),
    }
