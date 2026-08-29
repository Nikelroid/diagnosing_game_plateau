"""Three ways to put a number on how much a player cannot see, in bits.

All three answer the same question: at a typical decision, how many distinct world states are
consistent with what the player to move knows? Taking log2 of that count gives bits, so 0 means
perfect information and every extra bit doubles the set of worlds the player must reason over.

  exact       Walk the whole game tree, group histories by the acting player's information state,
              average log2 of each group's size. Ground truth, and only affordable on small games.
  closed form Count the ways the hidden hands could have been dealt. Exact arithmetic, but it
              counts hands only, so it ignores what public cards and betting have already ruled
              out. Read it as a mild overestimate.
  resampled   Draw worlds consistent with the information state and count the distinct ones.
              Censored at the number of samples, so it is a lower bound, and a badly saturated one
              looks exactly like a real measurement. Always compare against log2(n_resamples).
"""

from __future__ import annotations

import math
import random
import time

__all__ = ["hand_bits", "deal_bits", "exact_infoset_bits", "resampled_infoset_bits", "CENSOR_MARK"]


def hand_bits(unseen: int, k: int) -> float:
    """Bits needed to pin down which k of `unseen` cards one opponent holds."""
    if not 0 <= k <= unseen:
        return float("nan")
    return math.log2(math.comb(unseen, k))


def deal_bits(unseen: int, hands: list[int]) -> float:
    """Bits for several hidden hands at once: log2 of the ways to deal them from `unseen`.

    Multinomial rather than a single binomial, because at a four-handed table it matters which
    opponent holds which cards. Reduces to `hand_bits` when there is one hidden hand.
    """
    if sum(hands) > unseen:
        return float("nan")
    bits, pool = 0.0, unseen
    for k in hands:
        bits += math.log2(math.comb(pool, k))
        pool -= k
    return bits


def multiset_deal_bits(type_counts: list[tuple[int, int]], k: int) -> float:
    """Bits for one hidden hand of `k` drawn from a deck whose cards come in identical copies.

    `hand_bits` and `deal_bits` count C(unseen, k), which assumes every card is distinguishable.
    That is right for a 52-card deck and wrong for UNO and Mahjong, where a hand is a multiset:
    two red sevens are the same holding whichever physical card you were dealt. Counting them as
    distinct inflates the estimate, and by enough to put UNO above a ceiling it cannot reach.

    `type_counts` is a list of (copies of this type, how many types have that many copies), so
    UNO's 108 cards are [(1, 4), (2, 48), (4, 2)]. Returns log2 of the number of distinct hands.
    """
    poly = [1]
    for copies, count in type_counts:
        factor = [1] * (copies + 1)
        for _ in range(count):
            nxt = [0] * min(len(poly) + copies, k + 1)
            for i, a in enumerate(poly):
                if a:
                    for j, b in enumerate(factor):
                        if i + j <= k:
                            nxt[i + j] += a * b
            poly = nxt
    return math.log2(poly[k]) if len(poly) > k and poly[k] else float("nan")


def CENSOR_MARK(n_resamples: int) -> float:  # noqa: N802 - reads as a constant at call sites
    """The value a fully saturated resampled estimate returns. Any row at this number is a floor."""
    return math.log2(n_resamples)


def _infostate_key(state, player):
    """The best label available for what `player` can distinguish here."""
    try:
        return state.information_state_string(player)
    except Exception:  # noqa: BLE001
        try:
            return state.observation_string(player)
        except Exception:  # noqa: BLE001
            return None


def exact_infoset_bits(game, max_nodes=40_000, time_budget=12.0, max_chance_fanout=400):
    """Average and maximum log2(|infoset|) over reachable decision nodes, or None if too big.

    Worlds are keyed by `history_str()`, a compact action list rather than a rendered board, so the
    walk stays cheap. Three independent bail-outs, because one node cap is not enough: a node
    budget, a wall-clock budget, and a chance-fanout limit. The last matters most, since a game
    that deals a whole hand at one chance node enumerates millions of outcomes before anything else
    gets a chance to stop it.

    Returns (mean_bits, max_bits, nodes_walked).
    """
    groups: dict = {}
    nodes = [0]
    aborted = [False]
    deadline = time.time() + time_budget

    def walk(state):
        if aborted[0]:
            return
        nodes[0] += 1
        if nodes[0] > max_nodes or (nodes[0] % 512 == 0 and time.time() > deadline):
            aborted[0] = True
            return
        if state.is_terminal():
            return
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            if len(outcomes) > max_chance_fanout:
                aborted[0] = True
                return
            for action, _ in outcomes:
                walk(state.child(action))
            return
        if state.is_simultaneous_node():
            for player in range(game.num_players()):
                key = _infostate_key(state, player)
                if key is not None:
                    groups.setdefault((player, key), set()).add(state.history_str())
            for action in state.legal_actions(0):
                joint = [action] + [
                    state.legal_actions(p)[0] for p in range(1, game.num_players())
                ]
                child = state.clone()
                child.apply_actions(joint)
                walk(child)
            return
        player = state.current_player()
        key = _infostate_key(state, player)
        if key is not None:
            groups.setdefault((player, key), set()).add(state.history_str())
        for action in state.legal_actions():
            walk(state.child(action))

    try:
        walk(game.new_initial_state())
    except (RecursionError, MemoryError):
        return None, None, nodes[0]
    if aborted[0] or not groups:
        return None, None, nodes[0]
    sizes = [len(worlds) for worlds in groups.values()]
    mean = sum(math.log2(s) for s in sizes) / len(sizes)
    return mean, math.log2(max(sizes)), nodes[0]


def resampled_infoset_bits(game, n_points=12, n_resamples=80, seed=0, time_budget=10.0):
    """Lower bound for games too large to enumerate: (mean_bits, fraction_of_points_censored).

    A censored fraction near 1.0 means the estimator ran out of samples rather than out of worlds,
    and the number it returns is log2(n_resamples) regardless of the truth.
    """
    import pyspiel

    try:
        sampler = pyspiel.UniformProbabilitySampler(0.0, 1.0)
    except Exception:  # noqa: BLE001
        return None, None

    rng = random.Random(seed)
    deadline = time.time() + time_budget
    per_point, censored = [], 0

    for _ in range(n_points):
        if time.time() > deadline:
            break
        state = game.new_initial_state()
        target, steps = rng.randint(3, 12), 0
        while not state.is_terminal() and steps < target:
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(outcomes, probs)[0])
            elif state.is_simultaneous_node():
                state.apply_actions(
                    [rng.choice(state.legal_actions(p)) for p in range(game.num_players())]
                )
                steps += 1
            else:
                state.apply_action(rng.choice(state.legal_actions()))
                steps += 1
        if state.is_terminal() or state.is_chance_node():
            continue

        player = state.current_player()
        worlds = set()
        try:
            for _ in range(n_resamples):
                worlds.add(state.resample_from_infostate(player, sampler).history_str())
        except Exception:  # noqa: BLE001 - most games do not implement resampling
            return None, None
        if worlds:
            per_point.append(math.log2(len(worlds)))
            if len(worlds) > 0.9 * n_resamples:
                censored += 1

    if not per_point:
        return None, None
    return sum(per_point) / len(per_point), censored / len(per_point)
