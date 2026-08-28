"""Measure one OpenSpiel game: size, cost, and how much is hidden.

Everything here is time-boxed, because the survey spans games that differ by many orders of
magnitude in size and a single pathological call can otherwise stall the whole run. There are
three independent guards, and each one exists because a different game defeated the previous one:

  a node budget          for trees that are merely large
  a wall-clock budget    for trees that are large and slow to expand
  a SIGALRM alarm        for a single C++ call that never returns to Python, which the other two
                         guards cannot interrupt because they are only checked between iterations

The chance-fanout check must read `max_chance_outcomes()` rather than calling `chance_outcomes()`,
since a game that deals a whole hand at one chance node materializes every possible deal the
moment you ask, which hangs before any guard can fire.
"""

from __future__ import annotations

import random
import signal
import time

import pyspiel

from hidden_info_survey.catalog import GameSpec
from hidden_info_survey.hidden_bits import exact_infoset_bits, resampled_infoset_bits

__all__ = ["measure", "playout_stats"]

ALARM_SECONDS = 45
EXACT_TREE_LIMIT = 5e6  # rough estimate above which full enumeration is hopeless
CHANCE_FANOUT_LIMIT = 400


class _Timeout(Exception):
    pass


def _alarm(seconds):
    def _fire(signum, frame):
        raise _Timeout()

    signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)


def playout_stats(game, n_games, seed=0, time_budget=10.0):
    """Episode length and legal-action counts under uniform random play.

    Random play visits different states than a trained agent would, so these are a sense of scale
    rather than a cost model. Returns (mean_length, mean_branching, max_branching).
    """
    rng = random.Random(seed)
    deadline = time.time() + time_budget
    lengths, branches = [], []

    for _ in range(n_games):
        if lengths and time.time() > deadline:
            break
        state = game.new_initial_state()
        moves = 0
        while not state.is_terminal() and moves < 500:
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(outcomes, probs)[0])
            elif state.is_simultaneous_node():
                joint = []
                for player in range(game.num_players()):
                    legal = state.legal_actions(player)
                    branches.append(len(legal))
                    joint.append(rng.choice(legal))
                state.apply_actions(joint)
                moves += 1
            else:
                legal = state.legal_actions()
                branches.append(len(legal))
                state.apply_action(rng.choice(legal))
                moves += 1
        lengths.append(moves)

    return (
        sum(lengths) / max(1, len(lengths)),
        sum(branches) / max(1, len(branches)),
        max(branches) if branches else 0,
    )


def _shape_size(fn):
    try:
        shape = list(fn())
        size = 1
        for dim in shape:
            size *= dim
        return shape, size
    except Exception:  # noqa: BLE001 - plenty of games implement only one of the two tensors
        return None, None


def measure(spec: GameSpec, max_nodes=40_000, n_games=100) -> dict:
    """One row of the survey. Raises only if the game will not load at all."""
    started = time.time()
    game = pyspiel.load_game(spec.key)
    game_type = game.get_type()

    ist_shape, ist_size = _shape_size(game.information_state_tensor_shape)
    obs_shape, obs_size = _shape_size(game.observation_tensor_shape)

    try:
        _alarm(ALARM_SECONDS)
        mean_len, mean_branch, max_branch = playout_stats(game, n_games)
    except _Timeout:
        mean_len, mean_branch, max_branch = float("nan"), float("nan"), 0
    finally:
        signal.alarm(0)

    try:
        max_chance = game.max_chance_outcomes()
    except Exception:  # noqa: BLE001
        max_chance = 0

    estimated_tree = (
        float("inf") if mean_branch != mean_branch else mean_branch ** min(mean_len, 30)
    )

    exact_mean = exact_max = None
    nodes = 0
    if estimated_tree <= EXACT_TREE_LIMIT and max_chance <= CHANCE_FANOUT_LIMIT:
        try:
            _alarm(ALARM_SECONDS)
            exact_mean, exact_max, nodes = exact_infoset_bits(game, max_nodes)
        except _Timeout:
            pass
        finally:
            signal.alarm(0)

    resampled = censored = None
    if exact_mean is None:
        try:
            _alarm(ALARM_SECONDS)
            resampled, censored = resampled_infoset_bits(game)
        except _Timeout:
            pass
        finally:
            signal.alarm(0)

    def rounded(value, digits=2):
        return None if value is None else round(value, digits)

    return {
        "game": spec.key,
        "label": spec.label,
        "family": spec.family,
        "engine": "openspiel",
        "players": game.num_players(),
        "information": str(game_type.information).split(".")[-1],
        "dynamics": str(game_type.dynamics).split(".")[-1],
        "chance": str(game_type.chance_mode).split(".")[-1],
        "actions_declared": game.num_distinct_actions(),
        "branch_mean": None if mean_branch != mean_branch else round(mean_branch, 1),
        "branch_max": max_branch,
        "ist_shape": ist_shape,
        "obs_shape": obs_shape,
        "input_size": ist_size or obs_size,
        "maxlen_declared": game.max_game_length(),
        "len_mean": None if mean_len != mean_len else round(mean_len, 1),
        "bits_exact": rounded(exact_mean),
        "bits_exact_max": rounded(exact_max),
        "bits_resampled": rounded(resampled),
        "resample_censored_frac": rounded(censored),
        "bits_closed_form": rounded(spec.hidden, 1),
        "hidden_note": spec.note,
        "tree_nodes_walked": nodes,
        "seconds": round(time.time() - started, 1),
    }
