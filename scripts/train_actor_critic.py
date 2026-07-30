from __future__ import annotations

import argparse
import sys
import time
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
    TrainingRoundResult,
    learner_players_for_team,
    mean_training_result,
    train_actor_critic_round,
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
    "elapsed_seconds",
    "seed",
    "dealer",
    "train_total_loss",
    "train_policy_loss",
    "train_value_loss",
    "train_entropy",
    "train_mean_return",
    "train_steps",
    "train_learner_steps",
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
    parser.add_argument("--checkpoint", required=True, help="Where to save the trained model")
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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--entropy-bonus-weight", type=float, default=0.01)
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
        training_config = ActorCriticTrainingConfig(
            learner_players=learner_players,
            reward_mode=args.reward_mode,
            reward_scale=args.reward_scale,
            value_loss_weight=args.value_loss_weight,
            entropy_bonus_weight=args.entropy_bonus_weight,
            max_grad_norm=args.max_grad_norm,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    round_config = round_config_from_args(args)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    recent: list[TrainingRoundResult] = []
    started = time.perf_counter()
    print(
        "Training actor_critic "
        f"P{learner_players[0]}/P{learner_players[1]} vs {args.opponent} "
        f"for {args.rounds} rounds on {device}"
    )

    with MetricsWriter(
        jsonl_path=args.metrics_jsonl,
        csv_path=args.metrics_csv,
        csv_fields=METRIC_FIELDS,
    ) as metrics:
        for index in range(args.rounds):
            seed = args.seed + index
            dealer = (args.dealer + index) % 4
            try:
                result = train_actor_critic_round(
                    model,
                    optimizer,
                    baseline_agents,
                    seed=seed,
                    dealer=dealer,
                    round_config=round_config,
                    training_config=training_config,
                    generator=generator,
                )
            except ValueError as exc:
                parser.error(str(exc))
                return 2

            recent.append(result)
            round_number = index + 1
            elapsed_seconds = time.perf_counter() - started
            if round_number % args.log_every == 0 or round_number == args.rounds:
                means = _print_progress(round_number, args.rounds, recent)
                metrics.write(
                    _training_metrics_record(
                        round_number,
                        args.rounds,
                        recent,
                        means,
                        elapsed_seconds=elapsed_seconds,
                    )
                )
                recent.clear()

            if args.eval_every and (
                round_number % args.eval_every == 0 or round_number == args.rounds
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
                _print_evaluation(round_number, evaluation)
                metrics.write(
                    _evaluation_metrics_record(
                        round_number,
                        args.rounds,
                        evaluation,
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )

            if args.save_every and round_number % args.save_every == 0:
                save_actor_critic_checkpoint(checkpoint_path, model)

    save_actor_critic_checkpoint(checkpoint_path, model)
    print(f"Checkpoint written to {checkpoint_path}")
    return 0


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
    recent: list[TrainingRoundResult],
) -> dict[str, float]:
    means = mean_training_result(recent)
    last = recent[-1]
    print(
        f"round {round_number}/{total_rounds} "
        f"loss={means['total_loss']:+.4f} "
        f"policy={means['policy_loss']:+.4f} "
        f"value={means['value_loss']:.4f} "
        f"entropy={means['entropy']:.4f} "
        f"return={means['mean_return']:+.4f} "
        f"steps={means['steps']:.1f} "
        f"last_score={last.team_scores[0]}:{last.team_scores[1]}"
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
    recent: list[TrainingRoundResult],
    means: dict[str, float],
    *,
    elapsed_seconds: float,
) -> MetricRecord:
    last = recent[-1]
    record: dict[str, MetricValue] = {
        "event": "train",
        "round": round_number,
        "total_rounds": total_rounds,
        "elapsed_seconds": elapsed_seconds,
        "seed": last.seed,
        "dealer": last.dealer,
        "train_total_loss": means["total_loss"],
        "train_policy_loss": means["policy_loss"],
        "train_value_loss": means["value_loss"],
        "train_entropy": means["entropy"],
        "train_mean_return": means["mean_return"],
        "train_steps": means["steps"],
        "train_learner_steps": means["learner_steps"],
        "last_score_0": last.team_scores[0],
        "last_score_1": last.team_scores[1],
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
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.value_loss_weight < 0:
        parser.error("--value-loss-weight cannot be negative")
    if args.entropy_bonus_weight < 0:
        parser.error("--entropy-bonus-weight cannot be negative")
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


if __name__ == "__main__":
    raise SystemExit(main())
