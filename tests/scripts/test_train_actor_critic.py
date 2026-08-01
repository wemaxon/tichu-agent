from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml import ACTION_FEATURE_SIZE, OBSERVATION_FEATURE_SIZE
from ml.models import ActorCriticConfig, ParametricActorCritic
from ml.training import TrainingBatchResult
from scripts.benchmark_agents import BenchmarkResult, PlayedRound
from scripts.train_actor_critic import (
    CURRENT_POLICY_SPEC,
    LinearSchedule,
    TrainingCheckpointManager,
    _default_best_checkpoint_path,
    _evaluate_model,
    _evaluation_metrics_record,
    _training_metrics_record,
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


def _benchmark_result(score_diff: int) -> BenchmarkResult:
    return BenchmarkResult(
        agent_a=CURRENT_POLICY_SPEC,
        agent_b="greedy",
        round_pairs=1,
        seed=10_000,
        dealer=0,
        config=_fast_config(),
        confidence_level=0.95,
        rounds=(
            PlayedRound(
                index=0,
                seed=10_000,
                dealer=0,
                lineup=(CURRENT_POLICY_SPEC, "greedy", CURRENT_POLICY_SPEC, "greedy"),
                agent_a_team=0,
                team_scores=(score_diff, 0),
                finish_order=(0, 2, 1, 3),
                steps=10,
            ),
        ),
        elapsed_seconds=0.1,
    )


def test_default_best_checkpoint_path_uses_best_before_suffix() -> None:
    assert _default_best_checkpoint_path(Path("checkpoints/policy.pt")) == Path(
        "checkpoints/policy.best.pt"
    )
    assert _default_best_checkpoint_path(Path("checkpoints/policy")) == Path(
        "checkpoints/policy.best"
    )


def test_training_checkpoint_manager_keeps_latest_and_best(
    tmp_path: Path,
) -> None:
    manager = TrainingCheckpointManager(
        latest_path=tmp_path / "latest.pt",
        best_path=tmp_path / "best.pt",
        total_rounds=10,
        opponent="greedy",
        train_seed=0,
        dealer=0,
        learner_players=(0, 2),
    )
    model = _model()

    manager.save_latest(model, round_number=1)
    assert manager.latest_path.exists()
    latest = torch.load(manager.latest_path, weights_only=True)
    assert latest["metadata"]["checkpoint_role"] == "latest"
    assert latest["metadata"]["train_round"] == 1

    assert manager.maybe_save_best(model, 2, _benchmark_result(25))
    assert manager.best_path is not None
    best = torch.load(manager.best_path, weights_only=True)
    assert best["metadata"]["checkpoint_role"] == "best"
    assert best["metadata"]["train_round"] == 2
    assert best["metadata"]["eval_mean_score_diff"] == 25.0

    assert not manager.maybe_save_best(model, 3, _benchmark_result(5))
    unchanged = torch.load(manager.best_path, weights_only=True)
    assert unchanged["metadata"]["train_round"] == 2


def test_linear_schedule_interpolates_by_completed_rounds() -> None:
    schedule = LinearSchedule(start=0.02, end=0.002)

    assert schedule.value_at(0, 100) == pytest.approx(0.02)
    assert schedule.value_at(50, 100) == pytest.approx(0.011)
    assert schedule.value_at(100, 100) == pytest.approx(0.002)


def test_training_metrics_record_includes_batch_fields() -> None:
    result = TrainingBatchResult(
        first_seed=10,
        last_seed=13,
        first_dealer=0,
        last_dealer=3,
        round_count=4,
        steps=280,
        learner_steps=130,
        mean_reward=0.25,
        mean_score_diff=250.0,
        last_team_scores=(200, 0),
        total_loss=0.4,
        policy_loss=0.2,
        value_loss=0.5,
        entropy=1.1,
        mean_return=0.25,
        grad_norm=0.7,
        updated=True,
        entropy_bonus_weight=0.01,
        advantage_normalization="batch",
    )

    record = _training_metrics_record(
        16,
        100,
        2,
        13,
        [result],
        {
            "total_loss": result.total_loss,
            "policy_loss": result.policy_loss,
            "value_loss": result.value_loss,
            "entropy": result.entropy,
            "mean_return": result.mean_return,
            "mean_reward": result.mean_reward,
            "mean_score_diff": result.mean_score_diff,
            "steps": 70.0,
            "learner_steps": 32.5,
            "round_count": 4.0,
            "optimizer_updates": 1.0,
            "grad_norm": 0.7,
            "entropy_bonus_weight": 0.01,
        },
        elapsed_seconds=1.5,
    )

    assert record["event"] == "train"
    assert record["optimizer_step"] == 2
    assert record["total_optimizer_steps"] == 13
    assert record["train_rounds"] == 4
    assert record["train_batch_rounds"] == 4
    assert record["train_advantage_normalization"] == "batch"


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
