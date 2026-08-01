from __future__ import annotations

import torch

from agents.greedy import GreedyAgent
from ml import ACTION_FEATURE_SIZE, OBSERVATION_FEATURE_SIZE
from ml.models import ActorCriticConfig, ParametricActorCritic
from ml.training import (
    ActorCriticTrainingConfig,
    LearnerTransition,
    TrainingBatch,
    TrainingRollout,
    collect_training_batch,
    compute_actor_critic_loss,
    learner_players_for_team,
    train_actor_critic_batch,
)
from tichu.config import TichuRoundConfig


def _fast_config() -> TichuRoundConfig:
    return TichuRoundConfig(
        grand_tichu=False,
        small_tichu=False,
        card_exchange=False,
        bombs=False,
        wishes=False,
        special_cards=False,
    )


def _training_config() -> ActorCriticTrainingConfig:
    return ActorCriticTrainingConfig(
        learner_players=(0, 2),
        reward_mode="team_diff",
        reward_scale=1000.0,
        value_loss_weight=0.5,
        entropy_bonus_weight=0.01,
        advantage_normalization="batch",
        max_grad_norm=1.0,
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


def test_learner_players_for_team_returns_partners() -> None:
    assert learner_players_for_team(0) == (0, 2)
    assert learner_players_for_team(1) == (1, 3)


def test_collect_training_batch_records_only_learner_transitions() -> None:
    batch = collect_training_batch(
        _model(),
        {1: GreedyAgent(), 3: GreedyAgent()},
        seeds=(0, 1),
        dealers=(0, 1),
        round_config=_fast_config(),
        training_config=_training_config(),
    )

    assert batch.round_count == 2
    assert batch.steps > 0
    assert batch.transitions
    assert {transition.player for transition in batch.transitions} <= {0, 2}
    for rollout in batch.rollouts:
        assert rollout.rewards[0] == rollout.rewards[2]
        assert rollout.rewards[1] == rollout.rewards[3]


def test_compute_actor_critic_loss_normalizes_batch_advantages() -> None:
    batch = TrainingBatch(
        (
            TrainingRollout(
                seed=0,
                dealer=0,
                steps=2,
                transitions=(
                    LearnerTransition(
                        player=0,
                        action_index=0,
                        log_probability=torch.tensor(-1.0),
                        value=torch.tensor(0.0),
                        entropy=torch.tensor(0.2),
                    ),
                    LearnerTransition(
                        player=2,
                        action_index=1,
                        log_probability=torch.tensor(-2.0),
                        value=torch.tensor(0.0),
                        entropy=torch.tensor(0.2),
                    ),
                ),
                rewards=(1.0, -1.0, 1.0, -1.0),
                team_scores=(100, 0),
            ),
        )
    )

    unnormalized = compute_actor_critic_loss(
        batch,
        training_config=ActorCriticTrainingConfig(advantage_normalization="none"),
    )
    normalized = compute_actor_critic_loss(
        batch,
        training_config=ActorCriticTrainingConfig(advantage_normalization="batch"),
    )

    assert unnormalized.policy.item() > 0
    assert normalized.policy.item() == 0


def test_train_actor_critic_batch_updates_model_parameters() -> None:
    torch.manual_seed(0)
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    result = train_actor_critic_batch(
        model,
        optimizer,
        {1: GreedyAgent(), 3: GreedyAgent()},
        seeds=(1, 2),
        dealers=(0, 1),
        round_config=_fast_config(),
        training_config=_training_config(),
    )

    after = list(model.parameters())
    assert result.updated
    assert result.round_count == 2
    assert result.learner_steps > 0
    assert result.grad_norm is not None
    assert any(
        not torch.equal(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(before, after, strict=True)
    )
