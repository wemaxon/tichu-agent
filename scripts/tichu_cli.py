from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import Agent, available_agent_names, create_agent
from scripts.tichu_console import command_console
from tichu.config import TichuRoundConfig
from tichu.round import TichuRound


def play_round(
    seed: int,
    dealer: int,
    agents: list[Agent],
    *,
    verbose: bool = False,
    config: TichuRoundConfig | None = None,
) -> TichuRound:
    if len(agents) != 4:
        raise ValueError("Exactly four agents are required")

    round_ = TichuRound.new(seed=seed, dealer=dealer, config=config)
    steps = 0
    while not round_.terminal:
        player = round_.current_player
        assert player is not None
        action = agents[player].choose_action(round_)
        if verbose:
            print(f"{steps:04d} P{player} [{round_.phase.value}] {action}")
        round_ = round_.apply(action)
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


def _agent_names_from_args(args: argparse.Namespace) -> tuple[str, str, str, str]:
    if args.agents is not None:
        return tuple(args.agents)
    return (args.agent, args.agent, args.agent, args.agent)


def _build_agents(agent_names: tuple[str, str, str, str], seed: int) -> list[Agent]:
    return [
        create_agent(agent_name, seed * 17 + player)
        for player, agent_name in enumerate(agent_names)
    ]


def command_play(args: argparse.Namespace) -> int:
    config = _round_config_from_args(args)
    agent_names = _agent_names_from_args(args)
    aggregate = [0, 0]
    rounds: list[TichuRound] = []

    if args.verbose:
        print("Agents: " + " ".join(f"P{player}={name}" for player, name in enumerate(agent_names)))

    for offset in range(args.rounds):
        round_seed = args.seed + offset
        round_ = play_round(
            round_seed,
            (args.dealer + offset) % 4,
            _build_agents(agent_names, round_seed),
            verbose=args.verbose,
            config=config,
        )
        rounds.append(round_)
        assert round_.round_scores is not None
        aggregate[0] += round_.round_scores[0]
        aggregate[1] += round_.round_scores[1]
        print(
            f"Round {offset + 1}: finish={list(round_.finish_order)} "
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
    print(f"Finish order: {list(round_.finish_order)}")
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


def command_agents(_: argparse.Namespace) -> int:
    for name in available_agent_names():
        print(name)
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
    parser = argparse.ArgumentParser(description="Tichu simulation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play = subparsers.add_parser("play", help="Run one or more simulated rounds")
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--dealer", type=int, choices=range(4), default=0)
    play.add_argument("--rounds", type=int, default=1)
    play.add_argument("--verbose", action="store_true")
    play.add_argument("--log", type=str)
    play.add_argument(
        "--agent",
        choices=available_agent_names(),
        default="random",
        help="Agent used for all players unless --agents is provided",
    )
    play.add_argument(
        "--agents",
        choices=available_agent_names(),
        nargs=4,
        metavar=("P0", "P1", "P2", "P3"),
        help="Agents for players 0, 1, 2, and 3",
    )
    _add_round_config_args(play)
    play.set_defaults(func=command_play)

    console = subparsers.add_parser("console", help="Play one round from an interactive console")
    console.add_argument("--seed", type=int, default=0)
    console.add_argument("--dealer", type=int, choices=range(4), default=0)
    console.add_argument("--seat", type=int, choices=range(4), default=0)
    console.add_argument("--log", type=str)
    console.add_argument(
        "--agent",
        choices=available_agent_names(),
        default="random",
        help="Agent used for every non-human player unless --agents is provided",
    )
    console.add_argument(
        "--agents",
        choices=available_agent_names(),
        nargs=3,
        metavar=("A0", "A1", "A2"),
        help="Agents for the three non-human players in player order",
    )
    _add_round_config_args(console)
    console.set_defaults(func=command_console)

    replay = subparsers.add_parser("replay", help="Verify a saved replay")
    replay.add_argument("path")
    replay.set_defaults(func=command_replay)

    agents = subparsers.add_parser("agents", help="List available agent names")
    agents.set_defaults(func=command_agents)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
