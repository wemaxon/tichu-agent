from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

import torch
from torch import Tensor

from .encoding import (
    ACTION_FEATURE_SIZE,
    ACTION_SCHEMA_VERSION,
    OBSERVATION_FEATURE_SIZE,
    OBSERVATION_SCHEMA_VERSION,
)
from .models import ActorCriticConfig, ParametricActorCritic

CHECKPOINT_FORMAT_VERSION = 1
CheckpointPath: TypeAlias = str | Path


class CheckpointCompatibilityError(ValueError):
    """Raised when checkpoint metadata does not match the current ML schema."""


def save_actor_critic_checkpoint(
    path: CheckpointPath,
    model: ParametricActorCritic,
) -> None:
    _validate_feature_sizes(model.config)
    checkpoint = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "observation_feature_size": OBSERVATION_FEATURE_SIZE,
        "action_feature_size": ACTION_FEATURE_SIZE,
        "model_config": model.config.to_dict(),
        "model_state_dict": dict(model.state_dict()),
    }
    torch.save(checkpoint, Path(path))


def load_actor_critic_checkpoint(
    path: CheckpointPath,
    *,
    device: torch.device | str = "cpu",
) -> ParametricActorCritic:
    raw_checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=True,
    )
    checkpoint = _mapping(raw_checkpoint, "checkpoint")
    _require_version(
        checkpoint,
        "checkpoint_format_version",
        CHECKPOINT_FORMAT_VERSION,
    )
    _require_version(
        checkpoint,
        "observation_schema_version",
        OBSERVATION_SCHEMA_VERSION,
    )
    _require_version(
        checkpoint,
        "action_schema_version",
        ACTION_SCHEMA_VERSION,
    )
    _require_version(
        checkpoint,
        "observation_feature_size",
        OBSERVATION_FEATURE_SIZE,
    )
    _require_version(
        checkpoint,
        "action_feature_size",
        ACTION_FEATURE_SIZE,
    )

    config_values = _mapping(checkpoint.get("model_config"), "model_config")
    try:
        config = ActorCriticConfig.from_dict(config_values)
    except ValueError as error:
        raise CheckpointCompatibilityError(
            f"invalid checkpoint model config: {error}"
        ) from error
    _validate_feature_sizes(config)

    state_dict_values = _mapping(
        checkpoint.get("model_state_dict"),
        "model_state_dict",
    )
    if not all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in state_dict_values.items()
    ):
        raise CheckpointCompatibilityError(
            "model_state_dict must map parameter names to tensors"
        )

    model = ParametricActorCritic(config).to(device)
    try:
        model.load_state_dict(dict(state_dict_values), strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(
            f"checkpoint parameters do not match the model config: {error}"
        ) from error
    return model


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError(f"{name} must be a mapping")
    return value


def _require_version(
    checkpoint: Mapping[str, object],
    key: str,
    expected: int,
) -> None:
    actual = checkpoint.get(key)
    if isinstance(actual, bool) or not isinstance(actual, int):
        raise CheckpointCompatibilityError(f"{key} must be an integer")
    if actual != expected:
        raise CheckpointCompatibilityError(
            f"{key} is {actual}; expected {expected}"
        )


def _validate_feature_sizes(config: ActorCriticConfig) -> None:
    if config.observation_size != OBSERVATION_FEATURE_SIZE:
        raise CheckpointCompatibilityError(
            f"model observation size is {config.observation_size}; "
            f"current encoder size is {OBSERVATION_FEATURE_SIZE}"
        )
    if config.action_size != ACTION_FEATURE_SIZE:
        raise CheckpointCompatibilityError(
            f"model action size is {config.action_size}; "
            f"current encoder size is {ACTION_FEATURE_SIZE}"
        )
