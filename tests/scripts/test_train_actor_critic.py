from __future__ import annotations

from ml import ACTION_FEATURE_SIZE, OBSERVATION_FEATURE_SIZE
from ml.models import ActorCriticConfig, ParametricActorCritic
from scripts.train_actor_critic import (
    CURRENT_POLICY_SPEC,
    _evaluate_model,
    _evaluation_metrics_record,
)
from tichu.config import TichuRoundConfig


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


def _fast_config() -> TichuRoundConfig:
    return TichuRoundConfig(
        grand_tichu=False,
        small_tichu=False,
        card_exchange=False,
        bombs=False,
        wishes=False,
        special_cards=False,
    )


def test_evaluate_model_uses_current_in_memory_policy_and_restores_mode() -> None:
    model = _model()
    model.train()

    result = _evaluate_model(
        model,
        opponent="greedy",
        round_pairs=1,
        seed=10_000,
        dealer=0,
        config=_fast_config(),
        confidence_level=0.95,
    )

    assert model.training
    assert result.agent_a == CURRENT_POLICY_SPEC
    assert result.agent_b == "greedy"
    assert result.played_rounds == 2
    assert result.mean_score_diff_ci.confidence == 0.95


def test_evaluation_metrics_record_flattens_confidence_intervals() -> None:
    result = _evaluate_model(
        _model(),
        opponent="greedy",
        round_pairs=1,
        seed=20_000,
        dealer=0,
        config=_fast_config(),
        confidence_level=0.95,
    )

    record = _evaluation_metrics_record(
        5,
        10,
        result,
        elapsed_seconds=12.5,
    )

    assert record["event"] == "eval"
    assert record["round"] == 5
    assert record["eval_agent_a"] == CURRENT_POLICY_SPEC
    assert "eval_mean_score_diff_ci_low" in record
    assert "eval_agent_a_win_rate_ci_high" in record
