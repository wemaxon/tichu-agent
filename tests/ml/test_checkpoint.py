from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml import (
    ACTION_FEATURE_SIZE,
    OBSERVATION_FEATURE_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    ActorCriticConfig,
    CheckpointCompatibilityError,
    ParametricActorCritic,
    batch_candidates,
    load_actor_critic_checkpoint,
    save_actor_critic_checkpoint,
)


def _model() -> ParametricActorCritic:
    return ParametricActorCritic(
        ActorCriticConfig(
            observation_size=OBSERVATION_FEATURE_SIZE,
            action_size=ACTION_FEATURE_SIZE,
            state_hidden_size=16,
            action_hidden_size=12,
            embedding_size=8,
            scorer_hidden_size=6,
            value_hidden_size=4,
        )
    )


def _batch():
    return batch_candidates(
        observations=((0.0,) * OBSERVATION_FEATURE_SIZE,),
        action_groups=(
            (
                (0.0,) * ACTION_FEATURE_SIZE,
                (1.0,) * ACTION_FEATURE_SIZE,
            ),
        ),
    )


def test_checkpoint_round_trip_preserves_config_and_outputs(
    tmp_path: Path,
) -> None:
    torch.manual_seed(4)
    model = _model()
    path = tmp_path / "policy.pt"
    expected = model(_batch())

    save_actor_critic_checkpoint(path, model)
    restored = load_actor_critic_checkpoint(path)
    actual = restored(_batch())

    assert restored.config == model.config
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.values, expected.values)


def test_checkpoint_rejects_an_observation_schema_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incompatible.pt"
    save_actor_critic_checkpoint(path, _model())
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION + 1
    torch.save(checkpoint, path)

    with pytest.raises(
        CheckpointCompatibilityError,
        match="observation_schema_version",
    ):
        load_actor_critic_checkpoint(path)


def test_checkpoint_rejects_encoder_dimension_mismatch(tmp_path: Path) -> None:
    incompatible = ParametricActorCritic(
        ActorCriticConfig(
            observation_size=OBSERVATION_FEATURE_SIZE + 1,
            action_size=ACTION_FEATURE_SIZE,
        )
    )

    with pytest.raises(CheckpointCompatibilityError, match="observation size"):
        save_actor_critic_checkpoint(tmp_path / "incompatible.pt", incompatible)
