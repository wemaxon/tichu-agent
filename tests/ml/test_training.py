from __future__ import annotations

import torch

from agents.greedy import GreedyAgent
from ml import ACTION_FEATURE_SIZE, OBSERVATION_FEATURE_SIZE
from ml.models import ActorCriticConfig, ParametricActorCritic
from ml.training import (
    ActorCriticTrainingConfig,
    collect_training_round,
    learner_players_for_team,
    train_actor_critic_round,
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


def test_collect_training_round_records_only_learner_transitions() -> None:
    rollout = collect_training_round(
        _model(),
        {1: GreedyAgent(), 3: GreedyAgent()},
        seed=0,
        dealer=0,
        round_config=_fast_config(),
        training_config=_training_config(),
    )

    assert rollout.steps > 0
    assert rollout.transitions
    assert {transition.player for transition in rollout.transitions} <= {0, 2}
    assert rollout.rewards[0] == rollout.rewards[2]
    assert rollout.rewards[1] == rollout.rewards[3]


def test_train_actor_critic_round_updates_model_parameters() -> None:
    torch.manual_seed(0)
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    result = train_actor_critic_round(
        model,
        optimizer,
        {1: GreedyAgent(), 3: GreedyAgent()},
        seed=1,
        dealer=0,
        round_config=_fast_config(),
        training_config=_training_config(),
    )

    after = list(model.parameters())
    assert result.updated
    assert result.learner_steps > 0
    assert result.grad_norm is not None
    assert any(
        not torch.equal(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(before, after, strict=True)
    )
