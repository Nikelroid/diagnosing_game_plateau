"""The oracle probe with the same apparatus as the Gin Rummy experiment.

Why this exists
---------------
The tabular probe and the Gin Rummy probe differ on three axes at once: a table against a
neural network, a key multiplier against an extra channel on a fixed-width tensor, and a fixed
opponent against a self-play pool. Only one axis is the one under study. A control that shares
none of the apparatus cannot tell you whether the Gin Rummy null is about Gin Rummy or about
neural networks fed a wider tensor.

This runs the same three arms through a neural policy-gradient learner reading a fixed-width
information-state tensor, exactly as the Gin Rummy learner does. The oracle arm appends a
one-hot of the opponent's private card to that tensor, which is the tensor analogue of the fifth
plane. The placebo appends a one-hot drawn from the same support and independent of the deal.

    python scripts/probe_neural.py --game leduc_poker --arm oracle --seed 0
"""
from __future__ import annotations

import argparse, json, os, re, time
import numpy as np
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import cfr
from open_spiel.python.pytorch import policy_gradient

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def chance_history(state):
    out, replay = [], state.get_game().new_initial_state()
    for a in state.history():
        if replay.is_chance_node():
            out.append(a)
        replay.apply_action(a)
    return out


def extra_channel(state, player, arm, n_private, ep_rng):
    """The extra input the learner gets, as a one-hot over private outcomes."""
    v = np.zeros(n_private, dtype=np.float32)
    if arm == "baseline":
        return v
    dealt = chance_history(state)
    if arm == "oracle":
        opp = dealt[1 - player] if (1 - player) < len(dealt) else -1
    else:
        own = dealt[player] if player < len(dealt) else -1
        public = set(dealt[2:])
        allowed = [c for c in range(n_private) if c != own and c not in public]
        opp = allowed[ep_rng % len(allowed)] if allowed else -1
    if 0 <= opp < n_private:
        v[opp] = 1.0
    return v


def run(game_name, arm, seed, episodes, eval_games, cfr_iters, out_dir):
    t0 = time.time()
    game = pyspiel.load_game(game_name)
    n_private = max(2, len(game.new_initial_state().chance_outcomes()))

    solver = cfr.CFRSolver(game)
    for _ in range(cfr_iters):
        solver.evaluate_and_update_policy()
    ref = solver.average_policy()

    env = rl_environment.Environment(game_name)
    base_dim = env.observation_spec()["info_state"][0]
    dim = base_dim + (0 if arm == "baseline" else n_private)
    agent = policy_gradient.PolicyGradient(
        player_id=0, info_state_size=dim, num_actions=game.num_distinct_actions(),
        loss_str="rpg", hidden_layers_sizes=[64, 64], batch_size=32,
        entropy_cost=0.01, critic_learning_rate=0.01, pi_learning_rate=0.005)
    rng = np.random.default_rng(seed)

    def widen(ts, state, ep):
        obs = np.asarray(ts.observations["info_state"][0], dtype=np.float32)
        if arm == "baseline":
            return obs
        return np.concatenate([obs, extra_channel(state, 0, arm, n_private, ep)])

    def episode(learn):
        ts = env.reset(); ep = int(rng.integers(1 << 30)); total = 0.0
        while not env.is_turn_based or True:
            if ts.last():
                break
            p = ts.current_player()
            state = env.get_state
            if p == 0:
                ts.observations["info_state"][0] = list(widen(ts, state, ep))
                out = agent.step(ts, is_evaluation=not learn)
                ts = env.step([out.action])
            else:
                pr = ref.action_probabilities(state, p)
                acts = list(pr); w = np.array([pr[a] for a in acts], float)
                ts = env.step([int(rng.choice(acts, p=w / w.sum()))])
        ts.observations["info_state"][0] = list(np.zeros(dim, dtype=np.float32))
        if learn:
            agent.step(ts, is_evaluation=False)
        return ts.rewards[0]

    for _ in range(episodes):
        episode(True)
    scores = [episode(False) for _ in range(eval_games)]

    row = {"game": game_name, "arm": arm, "seed": seed, "episodes": episodes,
           "learner": "policy_gradient(rpg) 64x64", "input_dim": dim,
           "mean_return_vs_cfr": float(np.mean(scores)),
           "sem": float(np.std(scores) / np.sqrt(len(scores))),
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(out_dir, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", game_name).strip("_")
    with open(os.path.join(out_dir, f"{slug}_{arm}_s{seed}.json"), "w") as f:
        json.dump(row, f, indent=1)
    print(f"{game_name} {arm:8s} seed {seed} -> {row['mean_return_vs_cfr']:+.4f} "
          f"(dim {dim}, {row['seconds']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="leduc_poker")
    ap.add_argument("--arm", choices=("baseline", "oracle", "placebo"), default="oracle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=60_000)
    ap.add_argument("--eval-games", type=int, default=8_000)
    ap.add_argument("--cfr-iters", type=int, default=2_000)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "neural"))
    a = ap.parse_args()
    run(a.game, a.arm, a.seed, a.episodes, a.eval_games, a.cfr_iters, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
