"""Randomized complete-round engine invariant tests."""

from __future__ import annotations

import json

import pytest

from agents.random import RandomAgent
from tichu.actions import action_from_dict
from tichu.config import TichuRoundConfig
from tichu.round import Phase, TichuRound


pytestmark = [pytest.mark.random]


def _config_for_index(index: int) -> TichuRoundConfig:
    base_flags = index // 3
    bomb_mode = index % 3
    return TichuRoundConfig(
        grand_tichu=bool(base_flags & 1),
        small_tichu=bool(base_flags & 2),
        card_exchange=bool(base_flags & 4),
        bombs=bomb_mode != 0,
        out_of_turn_bombs=bomb_mode == 2,
        wishes=bool(base_flags & 8),
        special_cards=bool(base_flags & 16),
    )


def play_checked_round(seed: int, config: TichuRoundConfig) -> TichuRound:
    round_ = TichuRound.new(seed=seed, dealer=seed % 4, config=config)
    agents = [
        RandomAgent(
            seed * 17 + player,
            call_grand_probability=0.08,
            call_small_probability=0.08,
            bomb_probability=0.25,
        )
        for player in range(4)
    ]
    steps = 0
    while not round_.terminal:
        round_.assert_valid()
        player = round_.current_player
        assert player is not None, "Non-terminal round has no current player"
        legal_actions = round_.legal_actions()
        assert legal_actions, f"No legal actions in phase {round_.phase}"
        action = agents[player].choose_action(round_.observation(player), legal_actions)
        assert action in legal_actions, f"Agent chose illegal action: {action}"
        round_ = round_.apply(action)
        steps += 1
        assert steps <= 20_000, "Round exceeded the safety step limit"

    round_.assert_valid()
    assert round_.phase is Phase.ROUND_OVER
    assert round_.round_scores is not None
    assert len(round_.finish_order) in (2, 3)

    replayed = TichuRound.replay(
        seed=round_.seed,
        dealer=round_.dealer,
        config=round_.config,
        actions=[entry["action"] for entry in round_.action_log],
    )
    assert replayed.full_state() == round_.full_state()

    json.dumps(round_.full_state(), sort_keys=True)
    for entry in round_.action_log:
        action_from_dict(entry["action"])
    return round_


def test_random_rounds_preserve_engine_invariants(pytestconfig: pytest.Config) -> None:
    rounds = int(pytestconfig.getoption("--random-games"))
    for index in range(rounds):
        play_checked_round(index, _config_for_index(index % 96))
