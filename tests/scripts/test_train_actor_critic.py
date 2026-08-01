from __future__ import annotations

from pathlib import Path

import torch

from ml import ACTION_FEATURE_SIZE, OBSERVATION_FEATURE_SIZE
from ml.models import ActorCriticConfig, ParametricActorCritic
from scripts.benchmark_agents import BenchmarkResult, PlayedRound
from scripts.train_actor_critic import (
    CURRENT_POLICY_SPEC,
    TrainingCheckpointManager,
    _default_best_checkpoint_path,
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
