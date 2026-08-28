"""Prove the oracle plane is what it claims to be, before a single GPU-hour is spent on it.

A silently wrong oracle plane would not crash. It would train, plateau, and produce a confident
wrong answer to the question the paper asks, which is the worst failure mode available here. So
this checks the plane against the engine's ground truth over many seeds and many mid-game states,
and checks that the patch leaves nothing behind when it is switched off.

    python scripts/verify_oracle_plane.py [--seeds 30]
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np
from pettingzoo.classic import gin_rummy_v4

from oracle_probes.oracle_env import (
    disable,
    enable,
    is_enabled,
    opponent_hand_plane,
    oracle_observation,
)

BASE_PLANES = 4
DECK = 52


def engine_hands(env):
    """(acting player's hand, opponent's hand) straight from the engine, as sorted card strings."""
    game = env.unwrapped.env.game
    pid = game.round.current_player_id
    hands = game.round.players
    return (
        sorted(str(c) for c in hands[pid].hand),
        sorted(str(c) for c in hands[(pid + 1) % 2].hand),
    )


def check_one_episode(seed, rng, max_steps=40):
    """Walk one game with the oracle on, checking the extra plane at every decision point."""
    env = gin_rummy_v4.env(opponents_hand_visible=False)
    env.reset(seed=seed)

    checked = 0
    for step in range(max_steps):
        obs, reward, termination, truncation, _ = env.last()
        if termination or truncation:
            break
        observation = obs["observation"]

        assert observation.shape == (BASE_PLANES + 1, DECK), (
            f"expected {(BASE_PLANES + 1, DECK)} with the oracle on, got {observation.shape}"
        )

        # ground truth from the engine, encoded the same way plane 0 is
        raw = env.unwrapped.env
        pid = raw.game.round.current_player_id
        truth = np.asarray(raw._utils.encode_cards(raw.game.round.players[(pid + 1) % 2].hand))
        oracle = observation[BASE_PLANES]

        assert np.array_equal(oracle, truth), (
            f"seed {seed} step {step}: oracle plane does not match the opponent's hand"
        )
        # the opponent's hand is a real hand, and it is never also in ours
        assert 0 < oracle.sum() <= 11, f"seed {seed} step {step}: {oracle.sum()} cards in the plane"
        assert not np.any(np.logical_and(oracle, observation[0])), (
            f"seed {seed} step {step}: a card is in both hands"
        )
        # and it is genuinely new information: not equal to any plane the learner already had
        for p in range(BASE_PLANES):
            assert not np.array_equal(oracle, observation[p]), (
                f"seed {seed} step {step}: oracle plane duplicates plane {p}"
            )
        checked += 1

        mask = obs["action_mask"]
        legal = np.where(mask == 1)[0]
        env.step(int(rng.choice(legal)) if len(legal) else None)

    env.close()
    return checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    rng = random.Random(0)

    print("1. baseline observation is untouched when the patch is off")
    env = gin_rummy_v4.env()
    env.reset(seed=0)
    base_shape = env.last()[0]["observation"].shape
    env.close()
    assert base_shape == (BASE_PLANES, DECK), base_shape
    print(f"   ok: {base_shape}")

    print("2. the built-in flag is not an oracle (this is why the patch exists)")
    env = gin_rummy_v4.env(opponents_hand_visible=True)
    env.reset(seed=0)
    obs = env.last()[0]["observation"]
    raw = env.unwrapped.env
    pid = raw.game.round.current_player_id
    opp = raw.game.round.players[(pid + 1) % 2].hand
    builtin_fifth = obs[4].sum()
    env.close()
    print(f"   built-in fifth plane holds {int(builtin_fifth)} cards; the opponent holds {len(opp)}")
    assert builtin_fifth != len(opp), "the built-in flag now matches the hand; re-read this probe"

    print(f"3. oracle plane equals the opponent's hand, across {args.seeds} games")
    with oracle_observation():
        assert is_enabled()
        total = sum(check_one_episode(seed, rng) for seed in range(args.seeds))
    print(f"   ok: {total} decision points checked, all exact")

    print("4. the patch leaves nothing behind")
    assert not is_enabled()
    env = gin_rummy_v4.env()
    env.reset(seed=0)
    after = env.last()[0]["observation"].shape
    env.close()
    assert after == (BASE_PLANES, DECK), after
    print(f"   ok: back to {after}")

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
