"""Run the oracle-observation probe along the hidden-information axis.

Why this exists
---------------
The paper's central probe hands a learner the hidden state and reports that it gains nothing.
That null has no positive control. A reviewer is entitled to ask how we know the probe can
detect anything at all, and on Gin Rummy we cannot answer, because no optimum is computable
there.

Kuhn and Leduc poker fix that. Both are small enough that counterfactual regret minimisation
gives a true optimum, so for each learner we can measure two things the Gin Rummy study could
not: how far it sits from optimal play, and whether the oracle observation moves it.

The prediction that makes this a control: in a game the learner has already nearly solved,
extra information cannot help because there is no headroom. In a game the learner is far from
solving, extra information should help if the plateau is information-bound. Measuring the gap
to optimal separates "nothing left to gain" from "cannot use what it was given".

    python scripts/probe_axis.py --game kuhn_poker --episodes 200000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

import numpy as np
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import cfr, exploitability

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hidden_bits(game, cap=200_000):
    """Exact bits of hidden information at a typical decision.

    Walk the tree, group histories by the acting player's information-state string, and average
    log2 of each group's size. This is the same quantity the atlas reports, recomputed here so
    every probe result carries its own position on the axis rather than trusting a lookup.
    """
    from collections import defaultdict
    groups = defaultdict(int)
    stack, seen = [game.new_initial_state()], 0
    while stack:
        st = stack.pop()
        if st.is_terminal():
            continue
        if not st.is_chance_node():
            groups[(st.current_player(), st.information_state_string())] += 1
            seen += 1
            if seen > cap:
                return None
        acts = [o for o, _ in st.chance_outcomes()] if st.is_chance_node() else st.legal_actions()
        for a in acts:
            c = st.clone(); c.apply_action(a); stack.append(c)
    if not groups:
        return None
    return float(np.mean([np.log2(v) for v in groups.values()]))


def solve_cfr(game, iters):
    soc = cfr.CFRSolver(game)
    for _ in range(iters):
        soc.evaluate_and_update_policy()
    pol = soc.average_policy()
    return pol, exploitability.exploitability(game, pol)


_N_PRIVATE = 2          # set per game in main(); the number of distinct private outcomes


def chance_history(state):
    """Which entries of the history were dealt by chance, and their values.

    Replays the state so chance moves can be told apart from player actions. Needed because the
    placebo has to respect the same constraints the real card does: a public card that has been
    turned over can no longer be in the opponent's hand.
    """
    out, replay = [], state.get_game().new_initial_state()
    for a in state.history():
        if replay.is_chance_node():
            out.append(a)
        replay.apply_action(a)
    return out


def placebo_card(state, player, episode_rng_state):
    """A private card drawn from the same support as the real one, independent of the deal.

    The point of the placebo is to split the learner's table exactly as much as the oracle does
    while telling it nothing. That only works if it ranges over the same values: anything the
    opponent could still be holding, which excludes this player's own card and any card already
    face up. A token drawn from the full range instead pairs with states the real card never
    can, so it over-splits the table and overstates the cost of widening.
    """
    dealt = chance_history(state)
    own = dealt[player] if player < len(dealt) else -1
    public = set(dealt[2:])                      # anything chance revealed after the deal
    allowed = [c for c in range(_N_PRIVATE) if c != own and c not in public]
    if not allowed:
        return -1
    # a per-episode permutation, so the token is stable within an episode the way a real card is
    order = list(allowed)
    random.Random(episode_rng_state).shuffle(order)
    return order[0]


_DEALS_PER_PLAYER = 1   # set per game in main(); games deal more than one card or die


def private_card(state, player):
    """The opponent's private information, as the engine knows it.

    Read from the chance outcomes already dealt, never inferred, which is the same contract as
    the Gin Rummy plane. Games that deal several cards or dice per player return all of them:
    indexing the raw history by player number silently returns the wrong player's second die as
    soon as a game deals more than one, which is a bug that produces plausible numbers.
    """
    dealt = chance_history(state)
    k = _DEALS_PER_PLAYER
    got = dealt[player * k:(player + 1) * k]
    return got[0] if k == 1 and got else tuple(got)


def ceiling(game, opponent, oracle, player=0):
    """The best expected return `player` could get against this fixed opponent.

    With the opponent frozen the game is a decision problem, so this is exact. The oracle
    ceiling lets the player condition on the opponent's private card, which strictly refines its
    partition and raises the achievable value. Reporting both turns "the oracle arm scored
    higher" into two quantities that mean something: how much of the table the learner left
    (ceiling minus achieved) and what fraction of the information's worth it actually captured.
    """
    memo = {}

    def rec(state):
        # The tree revisits the same position by many action orders, so without a memo this
        # recursion is exponential and a 24k-state game never finishes.
        h = (tuple(state.history()),)
        if h in memo:
            return memo[h]
        v = _rec(state)
        memo[h] = v
        return v

    def _rec(state):
        if state.is_terminal():
            return state.returns()[player]
        if state.is_chance_node():
            return sum(p * rec_after(state, a) for a, p in state.chance_outcomes())
        if state.current_player() == player:
            return max(rec_after(state, a) for a in state.legal_actions())
        pr = opponent.action_probabilities(state, state.current_player())
        tot = sum(pr.values()) or 1.0
        return sum((w / tot) * rec_after(state, a) for a, w in pr.items())

    def rec_after(state, action):
        c = state.clone(); c.apply_action(action)
        return rec(c)

    if oracle:
        return rec(game.new_initial_state())
    from open_spiel.python.algorithms import best_response as br_lib
    return br_lib.BestResponsePolicy(game, player, opponent).value(game.new_initial_state())


def key(state, player, arm, rng=None):
    """The learner's observation key.

    oracle  appends the opponent's real private card.
    placebo appends a card drawn from the same distribution but independent of the deal. It
            splits the table exactly as much as the oracle does while carrying no information,
            which is what separates the value of the information from the cost of widening the
            observation to hold it.
    """
    k = state.information_state_string(player)
    if arm == "oracle":
        k += f"|opp={private_card(state, 1 - player)}"
    elif arm == "placebo":
        k += f"|opp={placebo_card(state, player, rng)}"
    return k


class Tabular:
    """Tabular Q-learning with epsilon-greedy exploration, identical in both arms."""

    def __init__(self, lr=0.10, eps=0.10, seed=0):
        self.q, self.lr, self.eps = {}, lr, eps
        self.rng = random.Random(seed)

    def act(self, k, legal, explore=True):
        row = self.q.setdefault(k, {a: 0.0 for a in legal})
        for a in legal:
            row.setdefault(a, 0.0)
        if explore and self.rng.random() < self.eps:
            return self.rng.choice(legal)
        best = max(legal, key=lambda a: row[a])
        return best

    def update(self, k, a, target):
        row = self.q.setdefault(k, {})
        row[a] = row.get(a, 0.0) + self.lr * (target - row.get(a, 0.0))


def train(game, episodes, arm, seed, opponent):
    """Train seat 0 against a fixed opponent policy. Returns the learner."""
    rng = np.random.default_rng(seed)
    agent = Tabular(seed=seed)
    for _ in range(episodes):
        state = game.new_initial_state()
        ep_token = int(rng.integers(1 << 30))     # one placebo draw per episode, as for a real deal
        trace = []
        while not state.is_terminal():
            if state.is_chance_node():
                outs, probs = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(outs, p=np.array(probs))))
                continue
            p = state.current_player()
            if p == 0:
                k = key(state, 0, arm, ep_token)
                a = agent.act(k, state.legal_actions())
                trace.append((k, a))
                state.apply_action(a)
            else:
                probs = opponent.action_probabilities(state, p)
                acts = list(probs); pr = np.array([probs[a] for a in acts], dtype=float)
                state.apply_action(int(rng.choice(acts, p=pr / pr.sum())))
        r = state.returns()[0]
        for k, a in trace:                       # Monte-Carlo return, same for both arms
            agent.update(k, a, r)
    return agent


def evaluate(game, agent, arm, opponent, games, seed):
    """Mean return for the learner in seat 0.

    Seats are NOT rotated. `train` only ever trains seat 0, and in these games the two seats
    have disjoint information-state strings, so a rotated evaluation asks the agent for keys it
    has never seen. Every such query hit a fresh all-zero row and broke the tie toward action 0,
    i.e. fold. That made half of every reported number a fold-bot, halving every effect and
    blending every level.
    """
    rng = np.random.default_rng(seed + 99991)
    total = 0.0
    for i in range(games):
        seat = 0
        state = game.new_initial_state()
        ep_token = int(rng.integers(1 << 30))
        while not state.is_terminal():
            if state.is_chance_node():
                outs, probs = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(outs, p=np.array(probs))))
                continue
            p = state.current_player()
            if p == seat:
                state.apply_action(agent.act(key(state, p, arm, ep_token), state.legal_actions(),
                                             explore=False))
            else:
                probs = opponent.action_probabilities(state, p)
                acts = list(probs); pr = np.array([probs[a] for a in acts], dtype=float)
                state.apply_action(int(rng.choice(acts, p=pr / pr.sum())))
        total += state.returns()[seat]
    return total / games


class AgentPolicy(policy_lib.Policy):
    """Wrap the tabular agent so OpenSpiel can measure its exploitability.

    Only defined for the non-oracle arm. An oracle agent conditions on information it would
    not hold in a real game, so its exploitability is not a well-posed quantity.
    """

    def __init__(self, game, agent):
        super().__init__(game, list(range(game.num_players())))
        self.agent = agent

    def action_probabilities(self, state, player_id=None):
        p = state.current_player() if player_id is None else player_id
        legal = state.legal_actions(p)
        a = self.agent.act(key(state, p, "baseline"), legal, explore=False)
        return {act: (1.0 if act == a else 0.0) for act in legal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="kuhn_poker")
    ap.add_argument("--episodes", type=int, default=200_000)
    ap.add_argument("--eval-games", type=int, default=20_000)
    ap.add_argument("--cfr-iters", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "axis"))
    args = ap.parse_args()

    t0 = time.time()
    game = pyspiel.load_game(args.game)
    bits = hidden_bits(game)
    print(f"[{args.game}] hidden information {bits if bits is None else round(bits, 3)} bits", flush=True)
    global _N_PRIVATE
    _N_PRIVATE = max(2, len(game.new_initial_state().chance_outcomes()))
    global _DEALS_PER_PLAYER
    probe, n_dealt = game.new_initial_state(), 0
    while probe.is_chance_node():
        probe.apply_action(probe.chance_outcomes()[0][0]); n_dealt += 1
    if n_dealt % game.num_players():
        raise SystemExit(f"{args.game}: {n_dealt} chance actions for {game.num_players()} "
                         "players is not a clean deal; the probe's assumption does not hold")
    _DEALS_PER_PLAYER = n_dealt // game.num_players()
    print(f"[{args.game}] {_DEALS_PER_PLAYER} private outcome(s) per player", flush=True)
    ref, expl_ref = solve_cfr(game, args.cfr_iters)
    print(f"[{args.game}] CFR reference exploitability {expl_ref:.5f}", flush=True)

    row = {"game": args.game, "seed": args.seed, "episodes": args.episodes,
           "hidden_bits": bits, "cfr_exploitability": expl_ref, "arms": {}}
    for arm in ("baseline", "oracle", "placebo"):
        oracle = arm == "oracle"
        name = arm
        agent = train(game, args.episodes, arm, args.seed, ref)
        ret = evaluate(game, agent, arm, ref, args.eval_games, args.seed)
        arm_states = len(agent.q)
        rec = {"mean_return_vs_cfr": ret, "states_visited": arm_states}
        rec["ceiling"] = ceiling(game, ref, oracle)
        rec["gap_to_ceiling"] = rec["ceiling"] - ret
        if arm == "baseline":
            rec["exploitability"] = exploitability.exploitability(game, AgentPolicy(game, agent))
        row["arms"][name] = rec
        print(f"  {name:9s} return {ret:+.4f}  ceiling {rec['ceiling']:+.4f} "
              f"gap {rec['gap_to_ceiling']:+.4f}  states {arm_states}", flush=True)

    ob = row["arms"]["oracle"]["mean_return_vs_cfr"] - row["arms"]["baseline"]["mean_return_vs_cfr"]
    worth = row["arms"]["oracle"]["ceiling"] - row["arms"]["baseline"]["ceiling"]
    pl = row["arms"]["placebo"]["mean_return_vs_cfr"]
    row["oracle_minus_baseline"] = ob
    row["oracle_minus_placebo"] = row["arms"]["oracle"]["mean_return_vs_cfr"] - pl
    row["widening_cost"] = row["arms"]["baseline"]["mean_return_vs_cfr"] - pl
    # The placebo still splits the table a little more than the oracle does, because an
    # independent draw pairs with states the real card never reaches. So it overstates the cost
    # of widening, and the true information effect is bracketed by the two differences above.
    row["placebo_oversplit"] = (row["arms"]["placebo"]["states_visited"]
                                - row["arms"]["oracle"]["states_visited"])
    row["information_worth"] = worth          # what the hidden card is worth against this opponent
    row["capture"] = (ob / worth) if abs(worth) > 1e-9 else None
    row["seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", args.game).strip("_")
    path = os.path.join(args.out, f"{slug}_s{args.seed}.json")
    json.dump(row, open(path, "w"), indent=1)
    print(f"  oracle minus baseline {row['oracle_minus_baseline']:+.4f}  ->  {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
