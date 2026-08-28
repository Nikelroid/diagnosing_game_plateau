"""Measure one RLCard game: the same row, from an engine that exposes far less.

RLCard has no information-state strings and no world resampling, so the exact and resampled
estimators simply do not apply here. What it does support is enough for everything else: action
count, observation size, episode length, and the empirical branching factor.

That difference is not a footnote. It means RLCard rows carry a closed form and measurements, and
never a tree-derived number, so they are marked with their engine everywhere they appear. The one
game measured by both engines, Gin Rummy, exists to check that the closed form agrees across
libraries; if that row ever disagrees, the arithmetic in the catalog is wrong.
"""

from __future__ import annotations

import random
import time

import rlcard

from hidden_info_survey.catalog import GameSpec

__all__ = ["measure", "playout_stats"]


def playout_stats(env, n_games, seed=0, time_budget=25.0):
    """Episode length and legal-action counts under uniform random play."""
    rng = random.Random(seed)
    deadline = time.time() + time_budget
    lengths, branches = [], []

    for _ in range(n_games):
        if lengths and time.time() > deadline:
            break
        state, _ = env.reset()
        moves = 0
        while not env.is_over() and moves < 400:
            legal = list(state["legal_actions"].keys())
            if not legal:
                break
            branches.append(len(legal))
            state, _ = env.step(rng.choice(legal))
            moves += 1
        lengths.append(moves)

    return (
        sum(lengths) / max(1, len(lengths)),
        sum(branches) / max(1, len(branches)),
        max(branches) if branches else 0,
        len(lengths),
    )


def _observation_size(env):
    try:
        shape = list(env.state_shape[0])
        size = 1
        for dim in shape:
            size *= dim
        return shape, size
    except Exception:  # noqa: BLE001
        return None, None


def measure(spec: GameSpec, n_games=60) -> dict:
    started = time.time()
    env = rlcard.make(spec.key, config={"seed": 0})
    obs_shape, obs_size = _observation_size(env)
    mean_len, mean_branch, max_branch, played = playout_stats(env, n_games)

    return {
        "game": f"rlcard:{spec.key}",
        "label": spec.label,
        "family": spec.family,
        "engine": "rlcard",
        "players": env.num_players,
        "information": "IMPERFECT_INFORMATION",
        "dynamics": "SEQUENTIAL",
        "chance": "EXPLICIT_STOCHASTIC",
        "actions_declared": env.num_actions,
        "branch_mean": round(mean_branch, 1),
        "branch_max": max_branch,
        "ist_shape": None,
        "obs_shape": obs_shape,
        "input_size": obs_size,
        "maxlen_declared": None,
        "len_mean": round(mean_len, 1),
        "bits_exact": None,  # no information-state strings in this engine
        "bits_exact_max": None,
        "bits_resampled": None,  # and no world resampling either
        "resample_censored_frac": None,
        "bits_closed_form": None if spec.hidden is None else round(spec.hidden, 1),
        "hidden_note": spec.note,
        "episodes_played": played,
        "tree_nodes_walked": 0,
        "seconds": round(time.time() - started, 1),
    }
