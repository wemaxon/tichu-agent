from __future__ import annotations

import pytest

from tichu.actions import PlayAction
from tichu.combinations import CombinationKind
from tichu.match import TichuMatch
from tichu.round import Phase

from .helpers import card_id, combo_of_kind, make_round


pytestmark = pytest.mark.scoring


def test_double_victory_scores_200_points() -> None:
    ace = combo_of_kind((card_id(14),), CombinationKind.SINGLE)
    round_ = make_round(
        {
            0: [],
            1: [card_id(2)],
            2: [card_id(14)],
            3: [card_id(3)],
        },
        current_player=2,
        finish_order=[0],
    )

    round_ = round_.apply(PlayAction(ace))

    assert round_.phase is Phase.ROUND_OVER
    assert round_.finish_order[:2] == (0, 2)
    assert round_.round_scores == (200, 0)


def test_final_remaining_players_cards_and_captured_tricks_are_assigned() -> None:
    last_play = combo_of_kind((card_id(2),), CombinationKind.SINGLE)
    tailender_hand = [card_id(5)]
    tailender_captured = [card_id(13)]
    round_ = make_round(
        {
            0: [],
            1: [],
            2: [card_id(2)],
            3: tailender_hand,
        },
        current_player=2,
        finish_order=[0, 1],
        captured={3: tailender_captured},
    )

    round_ = round_.apply(PlayAction(last_play))

    assert round_.phase is Phase.ROUND_OVER
    assert round_.hands[3] == ()
    assert round_.captured[3] == ()
    assert tailender_hand[0] in round_.captured[0]
    assert tailender_captured[0] in round_.captured[0]
    assert round_.round_scores == (15, 0)


def test_positive_and_negative_tichu_declarations_affect_scores() -> None:
    round_ = make_round(
        {
            0: [],
            1: [],
            2: [],
            3: [card_id(2)],
        },
        finish_order=[0, 2],
        tichu_calls={0: 100, 1: 100, 2: 200, 3: 200},
    )

    round_ = round_._finalize_double_victory()

    assert round_.round_scores == (100, -300)


def test_both_teams_crossing_target_score_ends_match_with_higher_score_winning() -> None:
    completed_round = make_round(
        {0: [], 1: [], 2: [], 3: []},
        phase=Phase.ROUND_OVER,
        current_player=None,
        finish_order=[0, 1, 2],
        round_scores=(20, 10),
    )
    match = TichuMatch(target_score=1000, scores=[990, 995])

    match = match.record_round(completed_round)

    assert match.terminal
    assert match.scores == (1010, 1005)
    assert match.winning_team == 0


def test_stalemate_over_target_keeps_match_alive_until_tie_breaks() -> None:
    completed_round = make_round(
        {0: [], 1: [], 2: [], 3: []},
        phase=Phase.ROUND_OVER,
        current_player=None,
        finish_order=[0, 1, 2],
        round_scores=(10, 10),
    )
    match = TichuMatch(target_score=1000, scores=[990, 990])

    match = match.record_round(completed_round)

    assert match.scores == (1000, 1000)
    assert not match.terminal
    assert match.winning_team is None
