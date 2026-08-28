"""Keep the oracle plane away from the opponents.

The probe is only meaningful if the oracle learner and the baseline learner differ in exactly one
respect. Widening the observation touches more than the learner, though: the curriculum plays the
learner against a pool of previously trained policies, and those policies read observations from
the same environment. A pool model trained on four planes cannot consume five, so without this
shim the run dies on a shape mismatch, and a careless fix (retraining the pool too) would change
the opponents as well as the observation and confound the result.

The shim trims each opponent's observation back to the number of planes that opponent's own
network expects. Opponents therefore see exactly what they saw during the baseline run, and the
extra plane reaches the learner alone. Agents that read raw planes directly, such as the
rule-based expert, index planes 0 and 1 and are unaffected either way.
"""

from __future__ import annotations

__all__ = ["enable_opponent_trim", "disable_opponent_trim", "is_enabled"]

_ORIGINAL_GET_OBSERVATION = None


def _trimmed_get_observation(self):
    obs = _ORIGINAL_GET_OBSERVATION(self)
    try:
        expected = self.model.observation_space["observation"].shape[0]
    except Exception:  # noqa: BLE001 - an agent without a Dict observation space needs no trim
        return obs
    observation = obs["observation"]
    if observation.shape[0] <= expected:
        return obs
    trimmed = dict(obs)
    trimmed["observation"] = observation[:expected]
    return trimmed


def is_enabled() -> bool:
    return _ORIGINAL_GET_OBSERVATION is not None


def enable_opponent_trim() -> None:
    """Patch PPO-backed opponents to ignore observation planes their network never saw."""
    global _ORIGINAL_GET_OBSERVATION
    if is_enabled():
        return
    from agents.ppo_agent import PPOAgent  # imported late: needs the main repo on sys.path

    _ORIGINAL_GET_OBSERVATION = PPOAgent.get_observation
    PPOAgent.get_observation = _trimmed_get_observation


def disable_opponent_trim() -> None:
    global _ORIGINAL_GET_OBSERVATION
    if not is_enabled():
        return
    from agents.ppo_agent import PPOAgent

    PPOAgent.get_observation = _ORIGINAL_GET_OBSERVATION
    _ORIGINAL_GET_OBSERVATION = None
