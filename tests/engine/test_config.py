"""Tests for round-rule configuration."""

from __future__ import annotations

import pytest

from scripts.round_runner import round_config_from_args
from scripts.tichu_cli import build_parser
from tichu.config import TichuRoundConfig


def test_bombs_default_to_normal_turn_timing() -> None:
    config = TichuRoundConfig()

    assert config.bombs
    assert not config.out_of_turn_bombs


def test_out_of_turn_bombs_require_bombs() -> None:
    with pytest.raises(ValueError, match="requires bombs"):
        TichuRoundConfig(bombs=False, out_of_turn_bombs=True)


def test_bomb_timing_survives_config_serialization() -> None:
    config = TichuRoundConfig(out_of_turn_bombs=True)

    assert TichuRoundConfig.from_dict(config.to_dict()) == config


def test_cli_enables_out_of_turn_bombs_explicitly() -> None:
    args = build_parser().parse_args(["play", "--out-of-turn-bombs"])

    config = round_config_from_args(args)

    assert config.bombs
    assert config.out_of_turn_bombs


def test_cli_rejects_conflicting_bomb_options() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["play", "--no-bombs", "--out-of-turn-bombs"],
        )
