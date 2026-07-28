from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.random import RandomAgent
from .config import TichuRoundConfig
from .round import TichuRound


def play_round(
    seed: int,
    dealer: int,
    verbose: bool = False,
    config: TichuRoundConfig | None = None,
) -> TichuRound:
    round_ = TichuRound.new(seed=seed, dealer=dealer, config=config)
    agents = [RandomAgent(seed * 17 + player, bomb_probability=0.15) for player in range(4)]
    steps = 0
    while not round_.terminal:
        player = round_.current_player
        assert player is not None
        action = agents[player].choose_action(round_)
        if verbose:
            print(f"{steps:04d} P{player} [{round_.phase.value}] {action}")
        round_.apply(action)
        steps += 1
        if steps > 20_000:
            raise RuntimeError("Round exceeded the safety step limit")
    return round_


def _round_config_from_args(args: argparse.Namespace) -> TichuRoundConfig:
    return TichuRoundConfig(
        grand_tichu=args.grand_tichu,
        small_tichu=args.small_tichu,
        card_exchange=args.card_exchange,
        bombs=args.bombs,
        wishes=args.wishes,
        special_cards=args.special_cards,
    )


def command_play(args: argparse.Namespace) -> int:
    config = _round_config_from_args(args)
    aggregate = [0, 0]
    rounds: list[TichuRound] = []
    for offset in range(args.rounds):
        round_ = play_round(
            args.seed + offset,
            (args.dealer + offset) % 4,
            args.verbose,
            config,
        )
        rounds.append(round_)
        assert round_.round_scores is not None
        aggregate[0] += round_.round_scores[0]
        aggregate[1] += round_.round_scores[1]
        print(
            f"Round {offset + 1}: finish={round_.finish_order} "
            f"score={round_.round_scores[0]}:{round_.round_scores[1]}"
        )
    print(f"Aggregate score: {aggregate[0]}:{aggregate[1]}")

    if args.log:
        if len(rounds) != 1:
            raise SystemExit("--log currently requires --rounds 1")
        rounds[0].save_replay(args.log)
        print(f"Replay written to {args.log}")
    return 0


def command_replay(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    round_ = TichuRound.replay(
        seed=int(payload["seed"]),
        dealer=int(payload.get("dealer", 0)),
        config=TichuRoundConfig.from_dict(payload.get("config")),
        actions=payload["actions"],
    )
    print(f"Finish order: {round_.finish_order}")
    print(f"Round score: {round_.round_scores}")
    expected = payload.get("final_state")
    if expected is not None:
        actual = round_.full_state()
        if "config" not in expected:
            actual = {key: value for key, value in actual.items() if key != "config"}
        if actual != expected:
            raise SystemExit("Replay completed, but the final state differs from the log")
    print("Replay verified")
    return 0


def _add_round_config_args(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        grand_tichu=True,
        small_tichu=True,
        card_exchange=True,
        bombs=True,
        wishes=True,
        special_cards=True,
    )
    parser.add_argument("--no-grand-tichu", "--no-big-tichu", dest="grand_tichu", action="store_false")
    parser.add_argument("--no-small-tichu", dest="small_tichu", action="store_false")
    parser.add_argument("--no-exchange", "--no-card-exchange", dest="card_exchange", action="store_false")
    parser.add_argument("--no-bombs", dest="bombs", action="store_false")
    parser.add_argument("--no-wishes", dest="wishes", action="store_false")
    parser.add_argument("--no-special-cards", dest="special_cards", action="store_false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic four-player Tichu simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play = subparsers.add_parser("play", help="Run random legal agents")
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--dealer", type=int, choices=range(4), default=0)
    play.add_argument("--rounds", type=int, default=1)
    play.add_argument("--verbose", action="store_true")
    play.add_argument("--log", type=str)
    _add_round_config_args(play)
    play.set_defaults(func=command_play)

    replay = subparsers.add_parser("replay", help="Verify a saved replay")
    replay.add_argument("path")
    replay.set_defaults(func=command_replay)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
