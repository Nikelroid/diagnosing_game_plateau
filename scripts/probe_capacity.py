"""The capacity probe, on games where the optimum is known.

The paper's capacity probe varies network size on Gin Rummy and finds no effect. That is a
null on one game with no way to check what "enough capacity" would even mean, because the
optimum is not computable there.

Here capacity is a knob with an exact meaning. The learner hashes each information state into
one of K buckets, so K is literally how many distinct situations it can tell apart. At K above
the game's information-state count the learner is exact; below it, states are forced to share
a policy. Sweeping K from starved to exact, against a CFR optimum, shows what a capacity-bound
plateau actually looks like, which is the comparison the Gin Rummy result lacks.

    python scripts/probe_capacity.py --game kuhn_poker --buckets 8 --seed 0
"""
from __future__ import annotations
import argparse, json, os, random, re, time
import numpy as np
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import cfr, exploitability

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bucket(key, k):
    """Fold an information state into one of k buckets. k<=0 means no folding."""
    if k <= 0:
        return key
    return f"b{hash(key) % k}"


class Tabular:
    def __init__(self, buckets, lr=0.10, eps=0.10, seed=0):
        self.q, self.lr, self.eps, self.k = {}, lr, eps, buckets
        self.rng = random.Random(seed)

    def act(self, key, legal, explore=True):
        k = bucket(key, self.k)
        row = self.q.setdefault(k, {})
        for a in legal:
            row.setdefault(a, 0.0)
        if explore and self.rng.random() < self.eps:
            return self.rng.choice(legal)
        return max(legal, key=lambda a: row[a])

    def update(self, key, a, target):
        row = self.q.setdefault(bucket(key, self.k), {})
        row[a] = row.get(a, 0.0) + self.lr * (target - row.get(a, 0.0))


class AgentPolicy(policy_lib.Policy):
    def __init__(self, game, agent):
        super().__init__(game, list(range(game.num_players())))
        self.agent = agent

    def action_probabilities(self, state, player_id=None):
        p = state.current_player() if player_id is None else player_id
        legal = state.legal_actions(p)
        a = self.agent.act(state.information_state_string(p), legal, explore=False)
        return {act: (1.0 if act == a else 0.0) for act in legal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="kuhn_poker")
    ap.add_argument("--buckets", type=int, default=0)      # 0 = full capacity
    ap.add_argument("--episodes", type=int, default=400_000)
    ap.add_argument("--eval-games", type=int, default=40_000)
    ap.add_argument("--cfr-iters", type=int, default=3_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "capacity"))
    args = ap.parse_args()

    t0 = time.time()
    game = pyspiel.load_game(args.game)
    solver = cfr.CFRSolver(game)
    for _ in range(args.cfr_iters):
        solver.evaluate_and_update_policy()
    ref = solver.average_policy()

    rng = np.random.default_rng(args.seed)
    agent = Tabular(args.buckets, seed=args.seed)
    for _ in range(args.episodes):
        state, trace = game.new_initial_state(), []
        while not state.is_terminal():
            if state.is_chance_node():
                outs, probs = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(outs, p=np.array(probs)))); continue
            p = state.current_player()
            if p == 0:
                k = state.information_state_string(0)
                a = agent.act(k, state.legal_actions())
                trace.append((k, a)); state.apply_action(a)
            else:
                pr = ref.action_probabilities(state, p)
                acts = list(pr); w = np.array([pr[a] for a in acts], float)
                state.apply_action(int(rng.choice(acts, p=w / w.sum())))
        r = state.returns()[0]
        for k, a in trace:
            agent.update(k, a, r)

    rng = np.random.default_rng(args.seed + 99991)
    total = 0.0
    for i in range(args.eval_games):
        # seat 0 only: train() never trains seat 1, and these games have seat-disjoint
        # information states, so a rotated evaluation scores an untrained fold-bot.
        seat, state = 0, game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outs, probs = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(outs, p=np.array(probs)))); continue
            p = state.current_player()
            if p == seat:
                state.apply_action(agent.act(state.information_state_string(p),
                                             state.legal_actions(), explore=False))
            else:
                pr = ref.action_probabilities(state, p)
                acts = list(pr); w = np.array([pr[a] for a in acts], float)
                state.apply_action(int(rng.choice(acts, p=w / w.sum())))
        total += state.returns()[seat]

    row = {"game": args.game, "buckets": args.buckets, "seed": args.seed,
           "episodes": args.episodes,
           "mean_return_vs_cfr": total / args.eval_games,
           "exploitability": exploitability.exploitability(game, AgentPolicy(game, agent)),
           "distinct_keys": len(agent.q), "seconds": round(time.time() - t0, 1)}
    os.makedirs(args.out, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", args.game).strip("_")
    json.dump(row, open(os.path.join(args.out, f"{slug}_k{args.buckets}_s{args.seed}.json"), "w"), indent=1)
    print(f"{args.game} buckets={args.buckets:5d} return {row['mean_return_vs_cfr']:+.4f} "
          f"expl {row['exploitability']:.4f} keys {row['distinct_keys']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
