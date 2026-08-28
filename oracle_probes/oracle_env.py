"""The oracle observation: give the learner the opponent's hand, change nothing else.

The probe this file implements
------------------------------
An agent that stops improving against a fixed reference is stuck for one of two reasons. Either
it cannot see enough to do better (information-bound), or it can see enough and fails to use it
(learning-bound). Retraining the identical recipe on an observation that contains the hidden
state separates the two: if the oracle learner does no better, the plateau was never about
information.

What gets added
---------------
PettingZoo's gin rummy gives the acting player four 52-card planes: own hand, top of the discard
pile, dead cards, and the opponent's known cards. This module appends a fifth plane holding the
opponent's *true* hand, taken from the engine rather than inferred.

Why not the built-in flag
-------------------------
`gin_rummy_v4.env(opponents_hand_visible=True)` sounds like this and is not. It exposes RLCard's
fifth plane, which is the *unknown* cards: the stock plus the unseen part of the opponent's hand.
That plane is almost the complement of your own hand, so it carries no hidden information at all.
Measured on a fresh deal it holds 41 cards, not the opponent's 10. The name is a trap.

Why a patch rather than a subclass
----------------------------------
The training stack builds its observation space by sampling one observation from the environment,
so widening the observation is picked up automatically and no trainer code has to change. That
keeps the oracle run and the baseline run on genuinely the same recipe, which is the entire point
of the probe. Patching is reversible and scoped: `disable()` restores the original methods, and
`oracle_observation()` is a context manager for the same thing.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo.classic.rlcard_envs import gin_rummy as _pz_gin

__all__ = ["opponent_hand_plane", "enable", "disable", "is_enabled", "oracle_observation"]

_ORIGINAL_OBSERVE = None
_ORIGINAL_OBSERVATION_SPACE = None


def opponent_hand_plane(rlcard_env, player_id: int) -> np.ndarray:
    """A 52-vector marking every card the *other* player is holding.

    The engine's own encoder is used, so this plane indexes cards exactly the way plane 0 does.
    Rolling our own card-to-column mapping here would be the easiest possible way to produce a
    silently misaligned oracle.
    """
    opponent = rlcard_env.game.round.players[(player_id + 1) % 2]
    return np.asarray(rlcard_env._utils.encode_cards(opponent.hand))


def _observe_with_oracle(self, agent):
    obs = _ORIGINAL_OBSERVE(self, agent)
    observation = obs["observation"]
    plane = opponent_hand_plane(self.env, self._name_to_int(agent))
    obs["observation"] = np.concatenate(
        [observation, plane[None, :].astype(observation.dtype)], axis=0
    )
    return obs


def _observation_space_with_oracle(self, agent):
    space = _ORIGINAL_OBSERVATION_SPACE(self, agent)
    box = space["observation"]
    return spaces.Dict(
        {
            "observation": spaces.Box(
                low=0, high=1, shape=(box.shape[0] + 1, box.shape[1]), dtype=box.dtype
            ),
            "action_mask": space["action_mask"],
        }
    )


def is_enabled() -> bool:
    return _ORIGINAL_OBSERVE is not None


def enable() -> None:
    """Make every gin rummy environment created from now on expose the oracle plane."""
    global _ORIGINAL_OBSERVE, _ORIGINAL_OBSERVATION_SPACE
    if is_enabled():
        return
    _ORIGINAL_OBSERVE = _pz_gin.raw_env.observe
    _ORIGINAL_OBSERVATION_SPACE = _pz_gin.raw_env.observation_space
    _pz_gin.raw_env.observe = _observe_with_oracle
    _pz_gin.raw_env.observation_space = _observation_space_with_oracle


def disable() -> None:
    """Restore the standard four-plane observation."""
    global _ORIGINAL_OBSERVE, _ORIGINAL_OBSERVATION_SPACE
    if not is_enabled():
        return
    _pz_gin.raw_env.observe = _ORIGINAL_OBSERVE
    _pz_gin.raw_env.observation_space = _ORIGINAL_OBSERVATION_SPACE
    _ORIGINAL_OBSERVE = None
    _ORIGINAL_OBSERVATION_SPACE = None


class oracle_observation:
    """`with oracle_observation():` for tests and for runs that should not leak the patch."""

    def __enter__(self):
        enable()
        return self

    def __exit__(self, *exc):
        disable()
        return False
