from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import Agent, available_agent_names, create_agent
from ml.checkpoint import load_actor_critic_checkpoint, save_actor_critic_checkpoint
from ml.policy import create_actor_critic
from ml.training import (
    ActorCriticTrainingConfig,
    TrainingRoundResult,
    learner_players_for_team,
    mean_training_result,
    train_actor_critic_round,
)
from scripts.round_runner import add_round_config_args, round_config_from_args


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
    print(
        "Training actor_critic "
        f"P{learner_players[0]}/P{learner_players[1]} vs {args.opponent} "
        f"for {args.rounds} rounds on {device}"
    )

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
        if round_number % args.log_every == 0 or round_number == args.rounds:
            _print_progress(round_number, args.rounds, recent)
            recent.clear()
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
) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
