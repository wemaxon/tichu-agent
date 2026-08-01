from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import (
    Agent,
    AgentCache,
    ActorCriticAgent,
    available_agent_names,
    create_agent,
)
from ml.checkpoint import load_actor_critic_checkpoint, save_actor_critic_checkpoint
from ml.models import ParametricActorCritic
from ml.policy import create_actor_critic
from ml.training import (
    ActorCriticTrainingConfig,
    AdvantageNormalization,
    TrainingBatchResult,
    learner_players_for_team,
    summarize_training_batches,
    train_actor_critic_batch,
)
from scripts.benchmark_agents import AgentLineup, BenchmarkResult, run_benchmark
from scripts.metrics import MetricRecord, MetricValue, MetricsWriter
from scripts.round_runner import add_round_config_args, round_config_from_args
from tichu.config import TichuRoundConfig


CURRENT_POLICY_SPEC = "actor_critic:current"

METRIC_FIELDS = (
    "event",
    "round",
    "total_rounds",
    "optimizer_step",
    "total_optimizer_steps",
    "elapsed_seconds",
    "seed",
    "dealer",
    "train_rounds",
    "train_batch_rounds",
    "train_optimizer_updates",
    "train_total_loss",
    "train_policy_loss",
    "train_value_loss",
    "train_entropy",
    "train_mean_return",
    "train_mean_reward",
    "train_mean_score_diff",
    "train_steps",
    "train_learner_steps",
    "train_grad_norm",
    "train_entropy_bonus_weight",
    "train_advantage_normalization",
    "last_score_0",
    "last_score_1",
    "eval_agent_a",
    "eval_agent_b",
    "eval_round_pairs",
    "eval_played_rounds",
    "eval_seed",
    "eval_score_diff",
    "eval_mean_score_diff",
    "eval_mean_score_diff_ci_low",
    "eval_mean_score_diff_ci_high",
    "eval_agent_a_win_rate",
    "eval_agent_a_win_rate_ci_low",
    "eval_agent_a_win_rate_ci_high",
    "eval_agent_a_wins",
    "eval_agent_b_wins",
    "eval_ties",
    "eval_agent_a_double_victories",
    "eval_agent_b_double_victories",
    "eval_mean_steps",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one shared actor-critic policy against fixed baseline agents"
    )
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Where to save the latest/final trained model",
    )
    parser.add_argument(
        "--best-checkpoint",
        help=(
            "Where to save the best held-out evaluation model. "
            "Requires --eval-every. Defaults to <checkpoint stem>.best<suffix>."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        help="Optional model checkpoint to initialize from; optimizer state is not loaded",
    )
    parser.add_argument(
        "--opponent",
        default="heuristic",
        help=(
            "Fixed baseline agent for the opposing team. "
            f"Available base names: {', '.join(available_agent_names())}. "
            "Use random, greedy, or heuristic for the first baseline runs."
        ),
    )
    parser.add_argument("--learner-team", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dealer", type=int, choices=range(4), default=0)
    parser.add_argument("--batch-rounds", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--entropy-bonus-weight", type=float, default=0.01)
    parser.add_argument(
        "--entropy-bonus-final-weight",
        type=float,
        help="Linearly decay entropy bonus to this value; omit to keep it constant",
    )
    parser.add_argument(
        "--advantage-normalization",
        choices=("none", "batch", "rollout"),
        default="batch",
        help="How to normalize policy advantages before the actor loss",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--reward-mode", choices=("team_diff", "team_score"), default="team_diff")
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run deterministic held-out evaluation every N rounds; 0 disables evaluation",
    )
    parser.add_argument(
        "--eval-round-pairs",
        type=int,
        default=20,
        help="Paired held-out round seeds per evaluation",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=1_000_000,
        help="First held-out seed used for deterministic evaluation",
    )
    parser.add_argument(
        "--eval-confidence",
        type=float,
        default=0.95,
        help="Confidence level for evaluation intervals",
    )
    parser.add_argument("--metrics-jsonl", help="Write training and eval metrics as JSONL")
    parser.add_argument("--metrics-csv", help="Write training and eval metrics as CSV")
    add_round_config_args(parser)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model = (
        load_actor_critic_checkpoint(args.init_checkpoint, device=device)
        if args.init_checkpoint
        else create_actor_critic(device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    learner_players = learner_players_for_team(args.learner_team)
    try:
        baseline_agents = _build_baseline_agents(
            args.opponent,
            learner_players=learner_players,
            seed=args.seed,
        )
        _training_config(
            args,
            learner_players=learner_players,
            entropy_bonus_weight=args.entropy_bonus_weight,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    round_config = round_config_from_args(args)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    checkpoint_manager = TrainingCheckpointManager(
        latest_path=Path(args.checkpoint),
        best_path=_best_checkpoint_path(
            Path(args.checkpoint),
            args.best_checkpoint,
            eval_enabled=bool(args.eval_every),
        ),
        total_rounds=args.rounds,
        opponent=args.opponent,
        train_seed=args.seed,
        dealer=args.dealer,
        learner_players=learner_players,
    )

    entropy_schedule = LinearSchedule(
        start=args.entropy_bonus_weight,
        end=(
            args.entropy_bonus_weight
            if args.entropy_bonus_final_weight is None
            else args.entropy_bonus_final_weight
        ),
    )
    total_optimizer_steps = math.ceil(args.rounds / args.batch_rounds)
    recent: list[TrainingBatchResult] = []
    started = time.perf_counter()
    print(
        "Training actor_critic "
        f"P{learner_players[0]}/P{learner_players[1]} vs {args.opponent} "
        f"for {args.rounds} rounds on {device} "
        f"({args.batch_rounds} rounds/update, {total_optimizer_steps} updates)"
    )
    print(
        "Optimizer: "
        f"Adam lr={args.learning_rate:g}, "
        f"advantage_normalization={args.advantage_normalization}, "
        f"entropy_bonus={entropy_schedule.start:g}->{entropy_schedule.end:g}"
    )
    print(f"Latest checkpoint: {checkpoint_manager.latest_path}")
    if checkpoint_manager.best_path is not None:
        print(f"Best checkpoint: {checkpoint_manager.best_path}")

    with MetricsWriter(
        jsonl_path=args.metrics_jsonl,
        csv_path=args.metrics_csv,
        csv_fields=METRIC_FIELDS,
    ) as metrics:
        completed_rounds = 0
        optimizer_step = 0
        next_log_round = args.log_every
        next_eval_round = args.eval_every if args.eval_every else None
        next_save_round = args.save_every if args.save_every else None

        while completed_rounds < args.rounds:
            batch_rounds = min(args.batch_rounds, args.rounds - completed_rounds)
            seeds = tuple(
                args.seed + completed_rounds + offset
                for offset in range(batch_rounds)
            )
            dealers = tuple(
                (args.dealer + completed_rounds + offset) % 4
                for offset in range(batch_rounds)
            )
            training_config = _training_config(
                args,
                learner_players=learner_players,
                entropy_bonus_weight=entropy_schedule.value_at(
                    completed_rounds,
                    args.rounds,
                ),
            )
            try:
                result = train_actor_critic_batch(
                    model,
                    optimizer,
                    baseline_agents,
                    seeds=seeds,
                    dealers=dealers,
                    round_config=round_config,
                    training_config=training_config,
                    generator=generator,
                )
            except ValueError as exc:
                parser.error(str(exc))
                return 2

            completed_rounds += batch_rounds
            optimizer_step += 1
            recent.append(result)
            elapsed_seconds = time.perf_counter() - started

            if completed_rounds >= next_log_round or completed_rounds == args.rounds:
                means = _print_progress(
                    completed_rounds,
                    args.rounds,
                    optimizer_step,
                    total_optimizer_steps,
                    recent,
                )
                metrics.write(
                    _training_metrics_record(
                        completed_rounds,
                        args.rounds,
                        optimizer_step,
                        total_optimizer_steps,
                        recent,
                        means,
                        elapsed_seconds=elapsed_seconds,
                    )
                )
                recent.clear()
                while next_log_round <= completed_rounds:
                    next_log_round += args.log_every

            if next_eval_round is not None and (
                completed_rounds >= next_eval_round
                or completed_rounds == args.rounds
            ):
                evaluation = _evaluate_model(
                    model,
                    opponent=args.opponent,
                    round_pairs=args.eval_round_pairs,
                    seed=args.eval_seed,
                    dealer=args.dealer,
                    config=round_config,
                    confidence_level=args.eval_confidence,
                )
                _print_evaluation(completed_rounds, evaluation)
                metrics.write(
                    _evaluation_metrics_record(
                        completed_rounds,
                        args.rounds,
                        evaluation,
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
                if checkpoint_manager.maybe_save_best(
                    model,
                    completed_rounds,
                    evaluation,
                ):
                    print(
                        "best checkpoint updated: "
                        f"round={completed_rounds} "
                        f"diff/round={evaluation.mean_score_diff:+.2f} "
                        f"path={checkpoint_manager.best_path}"
                    )
                while next_eval_round <= completed_rounds:
                    next_eval_round += args.eval_every

            if next_save_round is not None and completed_rounds >= next_save_round:
                checkpoint_manager.save_latest(model, completed_rounds)
                while next_save_round <= completed_rounds:
                    next_save_round += args.save_every

    checkpoint_manager.save_latest(model, args.rounds)
    print(f"Latest checkpoint written to {checkpoint_manager.latest_path}")
    if checkpoint_manager.best_path is not None:
        if checkpoint_manager.best_round is None:
            print("Best checkpoint was not written because no evaluation ran.")
        else:
            assert checkpoint_manager.best_score is not None
            print(
                "Best checkpoint: "
                f"{checkpoint_manager.best_path} "
                f"(round {checkpoint_manager.best_round}, "
                f"diff/round={checkpoint_manager.best_score:+.2f})"
            )
    return 0


@dataclass(slots=True)
class TrainingCheckpointManager:
    latest_path: Path
    best_path: Path | None
    total_rounds: int
    opponent: str
    train_seed: int
    dealer: int
    learner_players: tuple[int, ...]
    best_score: float | None = None
    best_round: int | None = None

    def save_latest(self, model: ParametricActorCritic, round_number: int) -> None:
        save_actor_critic_checkpoint(
            self.latest_path,
            model,
            metadata=self._metadata("latest", round_number),
        )

    def maybe_save_best(
        self,
        model: ParametricActorCritic,
        round_number: int,
        evaluation: BenchmarkResult,
    ) -> bool:
        if self.best_path is None:
            return False

        score = evaluation.mean_score_diff
        if self.best_score is not None and score <= self.best_score:
            return False

        self.best_score = score
        self.best_round = round_number
        save_actor_critic_checkpoint(
            self.best_path,
            model,
            metadata=self._metadata("best", round_number, evaluation),
        )
        return True

    def _metadata(
        self,
        role: str,
        round_number: int,
        evaluation: BenchmarkResult | None = None,
    ) -> dict[str, str | int | float | bool | None]:
        metadata: dict[str, str | int | float | bool | None] = {
            "checkpoint_role": role,
            "train_round": round_number,
            "total_rounds": self.total_rounds,
            "opponent": self.opponent,
            "train_seed": self.train_seed,
            "dealer": self.dealer,
            "learner_players": ",".join(str(player) for player in self.learner_players),
            "best_eval_round": self.best_round,
            "best_eval_mean_score_diff": self.best_score,
        }
        if evaluation is not None:
            metadata.update(
                {
                    "eval_round_pairs": evaluation.round_pairs,
                    "eval_played_rounds": evaluation.played_rounds,
                    "eval_seed": evaluation.seed,
                    "eval_mean_score_diff": evaluation.mean_score_diff,
                    "eval_agent_a_win_rate": evaluation.agent_a_win_rate,
                }
            )
        return metadata


@dataclass(frozen=True, slots=True)
class LinearSchedule:
    start: float
    end: float

    def value_at(self, completed_rounds: int, total_rounds: int) -> float:
        if total_rounds < 1:
            raise ValueError("total_rounds must be at least 1")
        progress = min(max(completed_rounds / total_rounds, 0.0), 1.0)
        return self.start + (self.end - self.start) * progress


def _build_baseline_agents(
    opponent: str,
    *,
    learner_players: tuple[int, ...],
    seed: int,
) -> dict[int, Agent]:
    learner_set = set(learner_players)
    return {
        player: create_agent(opponent, seed * 17 + player)
        for player in range(4)
        if player not in learner_set
    }


def _print_progress(
    round_number: int,
    total_rounds: int,
    optimizer_step: int,
    total_optimizer_steps: int,
    recent: list[TrainingBatchResult],
) -> dict[str, float]:
    means = summarize_training_batches(recent)
    last = recent[-1]
    print(
        f"round {round_number}/{total_rounds} "
        f"update {optimizer_step}/{total_optimizer_steps} "
        f"batch_rounds={int(means['round_count'])} "
        f"loss={means['total_loss']:+.4f} "
        f"policy={means['policy_loss']:+.4f} "
        f"value={means['value_loss']:.4f} "
        f"entropy={means['entropy']:.4f} "
        f"return={means['mean_return']:+.4f} "
        f"score_diff={means['mean_score_diff']:+.1f} "
        f"steps={means['steps']:.1f} "
        f"entropy_w={last.entropy_bonus_weight:.4f} "
        f"last_score={last.last_team_scores[0]}:{last.last_team_scores[1]}"
    )
    return means


def _evaluate_model(
    model: ParametricActorCritic,
    *,
    opponent: str,
    round_pairs: int,
    seed: int,
    dealer: int,
    config: TichuRoundConfig,
    confidence_level: float,
) -> BenchmarkResult:
    was_training = model.training
    model.eval()
    learner_agent = ActorCriticAgent.from_model(model, deterministic=True)
    agent_cache = AgentCache()

    def build_agents(lineup: AgentLineup, round_seed: int, offset: int) -> list[Agent]:
        agents: list[Agent] = []
        for player, spec in enumerate(lineup):
            if spec == CURRENT_POLICY_SPEC:
                agents.append(learner_agent)
            else:
                agents.append(
                    agent_cache.create(spec, round_seed * 10_000 + offset * 100 + player)
                )
        return agents

    try:
        return run_benchmark(
            agent_a=CURRENT_POLICY_SPEC,
            agent_b=opponent,
            round_pairs=round_pairs,
            seed=seed,
            dealer=dealer,
            config=config,
            confidence_level=confidence_level,
            build_agents=build_agents,
        )
    finally:
        model.train(was_training)


def _print_evaluation(round_number: int, result: BenchmarkResult) -> None:
    score_ci = result.mean_score_diff_ci
    win_ci = result.agent_a_win_rate_ci
    print(
        f"eval round {round_number}: "
        f"diff/round={result.mean_score_diff:+.2f} "
        f"ci={score_ci.low:+.2f}..{score_ci.high:+.2f} "
        f"win_rate={100 * result.agent_a_win_rate:.1f}% "
        f"ci={100 * win_ci.low:.1f}%..{100 * win_ci.high:.1f}% "
        f"score={result.agent_a_score}:{result.agent_b_score}"
    )


def _training_metrics_record(
    round_number: int,
    total_rounds: int,
    optimizer_step: int,
    total_optimizer_steps: int,
    recent: list[TrainingBatchResult],
    means: dict[str, float],
    *,
    elapsed_seconds: float,
) -> MetricRecord:
    last = recent[-1]
    record: dict[str, MetricValue] = {
        "event": "train",
        "round": round_number,
        "total_rounds": total_rounds,
        "optimizer_step": optimizer_step,
        "total_optimizer_steps": total_optimizer_steps,
        "elapsed_seconds": elapsed_seconds,
        "seed": last.last_seed,
        "dealer": last.last_dealer,
        "train_rounds": int(means["round_count"]),
        "train_batch_rounds": last.round_count,
        "train_optimizer_updates": int(means["optimizer_updates"]),
        "train_total_loss": means["total_loss"],
        "train_policy_loss": means["policy_loss"],
        "train_value_loss": means["value_loss"],
        "train_entropy": means["entropy"],
        "train_mean_return": means["mean_return"],
        "train_mean_reward": means["mean_reward"],
        "train_mean_score_diff": means["mean_score_diff"],
        "train_steps": means["steps"],
        "train_learner_steps": means["learner_steps"],
        "train_grad_norm": means["grad_norm"],
        "train_entropy_bonus_weight": means["entropy_bonus_weight"],
        "train_advantage_normalization": last.advantage_normalization,
        "last_score_0": last.last_team_scores[0],
        "last_score_1": last.last_team_scores[1],
    }
    return record


def _evaluation_metrics_record(
    round_number: int,
    total_rounds: int,
    result: BenchmarkResult,
    *,
    elapsed_seconds: float,
) -> MetricRecord:
    score_ci = result.mean_score_diff_ci
    win_ci = result.agent_a_win_rate_ci
    record: dict[str, MetricValue] = {
        "event": "eval",
        "round": round_number,
        "total_rounds": total_rounds,
        "elapsed_seconds": elapsed_seconds,
        "eval_agent_a": result.agent_a,
        "eval_agent_b": result.agent_b,
        "eval_round_pairs": result.round_pairs,
        "eval_played_rounds": result.played_rounds,
        "eval_seed": result.seed,
        "eval_score_diff": result.score_diff,
        "eval_mean_score_diff": result.mean_score_diff,
        "eval_mean_score_diff_ci_low": score_ci.low,
        "eval_mean_score_diff_ci_high": score_ci.high,
        "eval_agent_a_win_rate": result.agent_a_win_rate,
        "eval_agent_a_win_rate_ci_low": win_ci.low,
        "eval_agent_a_win_rate_ci_high": win_ci.high,
        "eval_agent_a_wins": result.agent_a_wins,
        "eval_agent_b_wins": result.agent_b_wins,
        "eval_ties": result.ties,
        "eval_agent_a_double_victories": result.agent_a_double_victories,
        "eval_agent_b_double_victories": result.agent_b_double_victories,
        "eval_mean_steps": result.mean_steps,
    }
    return record


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if args.batch_rounds < 1:
        parser.error("--batch-rounds must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.value_loss_weight < 0:
        parser.error("--value-loss-weight cannot be negative")
    if args.entropy_bonus_weight < 0:
        parser.error("--entropy-bonus-weight cannot be negative")
    if (
        args.entropy_bonus_final_weight is not None
        and args.entropy_bonus_final_weight < 0
    ):
        parser.error("--entropy-bonus-final-weight cannot be negative")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")
    if args.reward_scale <= 0:
        parser.error("--reward-scale must be positive")
    if args.save_every < 0:
        parser.error("--save-every cannot be negative")
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")
    if args.eval_every < 0:
        parser.error("--eval-every cannot be negative")
    if args.eval_round_pairs < 1:
        parser.error("--eval-round-pairs must be at least 1")
    if not 0.0 < args.eval_confidence < 1.0:
        parser.error("--eval-confidence must be between 0 and 1")
    if args.best_checkpoint and not args.eval_every:
        parser.error("--best-checkpoint requires --eval-every")
    if args.eval_every:
        latest_path = Path(args.checkpoint)
        best_path = _best_checkpoint_path(
            latest_path,
            args.best_checkpoint,
            eval_enabled=True,
        )
        assert best_path is not None
        if latest_path.expanduser().resolve() == best_path.expanduser().resolve():
            parser.error("--best-checkpoint must differ from --checkpoint")


def _training_config(
    args: argparse.Namespace,
    *,
    learner_players: tuple[int, ...],
    entropy_bonus_weight: float,
) -> ActorCriticTrainingConfig:
    return ActorCriticTrainingConfig(
        learner_players=learner_players,
        reward_mode=args.reward_mode,
        reward_scale=args.reward_scale,
        value_loss_weight=args.value_loss_weight,
        entropy_bonus_weight=entropy_bonus_weight,
        advantage_normalization=_advantage_normalization(args.advantage_normalization),
        max_grad_norm=args.max_grad_norm,
    )


def _advantage_normalization(value: object) -> AdvantageNormalization:
    if value == "none":
        return "none"
    if value == "batch":
        return "batch"
    if value == "rollout":
        return "rollout"
    raise ValueError(
        "advantage_normalization must be 'none', 'batch', or 'rollout'"
    )


def _best_checkpoint_path(
    latest_path: Path,
    requested_path: str | None,
    *,
    eval_enabled: bool,
) -> Path | None:
    if requested_path is not None:
        return Path(requested_path)
    if not eval_enabled:
        return None
    return _default_best_checkpoint_path(latest_path)


def _default_best_checkpoint_path(latest_path: Path) -> Path:
    if latest_path.suffix:
        return latest_path.with_name(f"{latest_path.stem}.best{latest_path.suffix}")
    return latest_path.with_name(f"{latest_path.name}.best")


if __name__ == "__main__":
    raise SystemExit(main())
