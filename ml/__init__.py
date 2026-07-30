from __future__ import annotations

from importlib import import_module
from typing import Any

from .policy_view import (
    PLAYER_COUNT,
    POLICY_CONFIG_KEYS,
    PolicyCombination,
    PolicyExchange,
    PolicyExchangeCard,
    PolicyObservation,
    absolute_player,
    policy_view,
    relative_player,
    rotate_player_values,
)


_EXPORT_MODULES = {
    "CandidateBatch": ".batching",
    "EncodedCandidate": ".interfaces",
    "EncodedPolicyState": ".interfaces",
    "InferencePolicy": ".interfaces",
    "PolicyDecisionLike": ".interfaces",
    "batch_candidates": ".batching",
    "batch_env_states": ".batching",
    "CHECKPOINT_FORMAT_VERSION": ".checkpoint",
    "CheckpointCompatibilityError": ".checkpoint",
    "load_actor_critic_checkpoint": ".checkpoint",
    "save_actor_critic_checkpoint": ".checkpoint",
    "RaggedCategorical": ".distributions",
    "ACTION_FEATURE_SIZE": ".encoding",
    "ACTION_SCHEMA_VERSION": ".encoding",
    "ACTION_TYPES": ".encoding",
    "COMBINATION_FEATURE_SIZE": ".encoding",
    "COMBINATION_KINDS": ".encoding",
    "CONFIG_KEYS": ".encoding",
    "OBSERVATION_FEATURE_SIZE": ".encoding",
    "OBSERVATION_SCHEMA_VERSION": ".encoding",
    "PHASES": ".encoding",
    "EncodedAction": ".encoding",
    "encode_action": ".encoding",
    "encode_card_mask": ".encoding",
    "encode_legal_actions": ".encoding",
    "encode_observation": ".encoding",
    "encode_policy_observation": ".encoding",
    "PlayerRewards": ".env",
    "RewardMode": ".env",
    "TeamRewards": ".env",
    "TichuEnv": ".env",
    "TichuEnvState": ".env",
    "encode_env_state": ".env",
    "ActorCriticConfig": ".models",
    "ActorCriticOutput": ".models",
    "ParametricActorCritic": ".models",
    "ParametricPolicy": ".policy",
    "PolicyDecision": ".policy",
    "create_actor_critic": ".policy",
    "default_actor_critic_config": ".policy",
    "PLAYER_COUNT": ".policy_view",
    "POLICY_CONFIG_KEYS": ".policy_view",
    "PolicyCombination": ".policy_view",
    "PolicyExchange": ".policy_view",
    "PolicyExchangeCard": ".policy_view",
    "PolicyObservation": ".policy_view",
    "absolute_player": ".policy_view",
    "policy_view": ".policy_view",
    "relative_player": ".policy_view",
    "rotate_player_values": ".policy_view",
}

__all__ = tuple(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
