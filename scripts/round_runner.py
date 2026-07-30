from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from agents import Agent
from tichu.config import TichuRoundConfig
from tichu.round import TichuRound


@dataclass(frozen=True, slots=True)
class AgentRoundResult:
    round: TichuRound
    steps: int


def play_agent_round(
    *,
    seed: int,
    dealer: int,
    agents: Sequence[Agent],
    config: TichuRoundConfig | None = None,
    verbose: bool = False,
    safety_limit: int = 20_000,
) -> AgentRoundResult:
    if len(agents) != 4:
        raise ValueError("Exactly four agents are required")

    round_ = TichuRound.new(seed=seed, dealer=dealer, config=config)
    steps = 0
    while not round_.terminal:
        player = round_.current_player
        if player is None:
            raise RuntimeError("Non-terminal round has no current player")

        legal_actions = round_.legal_actions()
        if not legal_actions:
            raise RuntimeError(f"No legal actions in phase {round_.phase.value}")
        action = agents[player].choose_action(round_.observation(player), legal_actions)
        if action not in legal_actions:
            raise RuntimeError(f"Agent at P{player} chose an illegal action: {action!r}")

        if verbose:
            print(f"{steps:04d} P{player} [{round_.phase.value}] {action}")
        round_ = round_.apply(action, legal_actions=legal_actions)
        steps += 1
        if steps > safety_limit:
            raise RuntimeError("Round exceeded the safety step limit")

    return AgentRoundResult(round=round_, steps=steps)


def round_config_from_args(args: argparse.Namespace) -> TichuRoundConfig:
    return TichuRoundConfig(
        grand_tichu=args.grand_tichu,
        small_tichu=args.small_tichu,
        card_exchange=args.card_exchange,
        bombs=args.bombs,
        out_of_turn_bombs=args.out_of_turn_bombs,
        wishes=args.wishes,
        special_cards=args.special_cards,
    )


def add_round_config_args(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        grand_tichu=True,
        small_tichu=True,
        card_exchange=True,
        bombs=True,
        out_of_turn_bombs=False,
        wishes=True,
        special_cards=True,
    )
    parser.add_argument("--no-grand-tichu", "--no-big-tichu", dest="grand_tichu", action="store_false")
    parser.add_argument("--no-small-tichu", dest="small_tichu", action="store_false")
    parser.add_argument("--no-exchange", "--no-card-exchange", dest="card_exchange", action="store_false")
    bomb_options = parser.add_mutually_exclusive_group()
    bomb_options.add_argument("--no-bombs", dest="bombs", action="store_false")
    bomb_options.add_argument(
        "--out-of-turn-bombs",
        dest="out_of_turn_bombs",
        action="store_true",
        help="Allow bombs to interrupt play outside the holder's normal turn",
    )
    parser.add_argument("--no-wishes", dest="wishes", action="store_false")
    parser.add_argument("--no-special-cards", dest="special_cards", action="store_false")
