"""Tests for immutable engine transitions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tichu.actions import Action, GrandTichuDecision
from tichu.round import TichuRound


def test_round_apply_returns_new_state_and_leaves_original_unchanged() -> None:
    round_ = TichuRound.new(seed=0)
    before = round_.full_state()

    next_round = round_.apply(GrandTichuDecision(False))

    assert next_round is not round_
    assert round_.full_state() == before
    assert round_.action_log == ()
    assert len(next_round.action_log) == 1


def test_round_apply_can_reuse_precomputed_legal_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    round_ = TichuRound.new(seed=0)
    legal_actions = round_.legal_actions()

    def fail_legal_actions(self: TichuRound) -> tuple[Action, ...]:
        raise AssertionError("legal_actions should not be recomputed")

    monkeypatch.setattr(TichuRound, "legal_actions", fail_legal_actions)

    next_round = round_.apply(GrandTichuDecision(False), legal_actions=legal_actions)

    assert next_round is not round_
    assert len(next_round.action_log) == 1


def test_round_state_fields_are_immutable() -> None:
    round_ = TichuRound.new(seed=0)

    with pytest.raises(FrozenInstanceError):
        round_.current_player = 1

    with pytest.raises(TypeError):
        round_.hands[0][0] = round_.hands[0][0]

    with pytest.raises(AttributeError):
        round_.discarded.append(1)
