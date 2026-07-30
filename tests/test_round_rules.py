from __future__ import annotations

import pytest

from tichu.actions import BombAction, DragonGiftAction, PassAction, PlayAction
from tichu.cards import Special, Suit
from tichu.combinations import CombinationKind
from tichu.config import TichuRoundConfig
from tichu.round import BombCloseBehavior, Phase

from .helpers import card_id, combo_of_kind, make_round, normal_ids


pytestmark = pytest.mark.rules


def _out_of_turn_bomb_config() -> TichuRoundConfig:
    return TichuRoundConfig(out_of_turn_bombs=True)


def test_mahjong_wish_forces_fulfillable_plays() -> None:
    mahjong = combo_of_kind((card_id(special=Special.MAHJONG),), CombinationKind.SINGLE)
    round_ = make_round(
        {
            0: [card_id(special=Special.MAHJONG)],
            1: [card_id(5), card_id(9)],
            2: [card_id(2)],
            3: [card_id(3)],
        },
        current_player=0,
    )

    round_ = round_.apply(PlayAction(mahjong, 5))

    assert round_.wish_rank == 5
    assert round_.current_player == 1
    actions = round_.legal_actions()
    play_actions = [action for action in actions if isinstance(action, PlayAction)]
    assert play_actions
    assert all(action.combination.contains_normal_rank(5) for action in play_actions)
    assert not any(isinstance(action, PassAction) for action in actions)


def test_mahjong_wish_does_not_force_unfulfillable_plays() -> None:
    top = combo_of_kind((card_id(4),), CombinationKind.SINGLE)
    round_ = make_round(
        {
            0: [],
            1: [card_id(6), card_id(9)],
            2: [card_id(2)],
            3: [card_id(3)],
        },
        current_player=1,
        wish_rank=5,
        top_combination=top,
        trick_cards=top.card_ids,
        last_play_player=0,
    )

    actions = round_.legal_actions()

    assert any(isinstance(action, PassAction) for action in actions)
    assert any(
        isinstance(action, PlayAction)
        and not action.combination.contains_normal_rank(5)
        for action in actions
    )


def test_wish_is_fulfilled_through_a_combination() -> None:
    top_pair = combo_of_kind(
        (card_id(4, Suit.JADE), card_id(4, Suit.SWORD)),
        CombinationKind.PAIR,
    )
    wished_pair = combo_of_kind(
        (card_id(5, Suit.JADE), card_id(5, Suit.SWORD)),
        CombinationKind.PAIR,
    )
    round_ = make_round(
        {
            0: [],
            1: list(wished_pair.card_ids),
            2: [card_id(2)],
            3: [card_id(3)],
        },
        current_player=1,
        wish_rank=5,
        top_combination=top_pair,
        trick_cards=top_pair.card_ids,
        last_play_player=0,
    )

    round_ = round_.apply(PlayAction(wished_pair))

    assert round_.wish_rank is None
    assert {"type": "wish_fulfilled", "rank": 5, "player": 1} in round_.public_history


def test_wish_can_be_fulfilled_during_bomb_interruption() -> None:
    top = combo_of_kind((card_id(10),), CombinationKind.SINGLE)
    bomb = combo_of_kind(normal_ids(5), CombinationKind.FOUR_BOMB)
    round_ = make_round(
        {
            0: [],
            1: list(bomb.card_ids),
            2: [card_id(2)],
            3: [card_id(3)],
        },
        config=_out_of_turn_bomb_config(),
        phase=Phase.BOMB_WINDOW,
        current_player=1,
        wish_rank=5,
        top_combination=top,
        trick_cards=top.card_ids,
        last_play_player=0,
        bomb_queue=[1],
        bomb_resume_player=2,
        bomb_close_behavior=BombCloseBehavior.CONTINUE,
    )

    round_ = round_.apply(BombAction(bomb))

    assert round_.wish_rank is None
    assert {"type": "wish_fulfilled", "rank": 5, "player": 1} in round_.public_history


def test_dog_transfers_lead_to_partner() -> None:
    dog = combo_of_kind((card_id(special=Special.DOG),), CombinationKind.DOG)
    round_ = make_round(
        {
            0: [card_id(special=Special.DOG)],
            1: [card_id(2)],
            2: [card_id(3)],
            3: [card_id(4)],
        },
        current_player=0,
    )

    round_ = round_.apply(PlayAction(dog))

    assert round_.current_player == 2
    assert 0 in round_.finish_order


def test_dog_skips_partner_who_has_already_gone_out() -> None:
    dog = combo_of_kind((card_id(special=Special.DOG),), CombinationKind.DOG)
    round_ = make_round(
        {
            0: [card_id(special=Special.DOG), card_id(9)],
            1: [card_id(2)],
            2: [],
            3: [card_id(4)],
        },
        current_player=0,
        finish_order=[2],
    )

    round_ = round_.apply(PlayAction(dog))

    assert round_.current_player == 3


def test_dragon_recipient_selection_and_capture() -> None:
    trick_cards = [card_id(special=Special.DRAGON), card_id(14)]
    round_ = make_round(
        {0: [], 1: [card_id(2)], 2: [card_id(3)], 3: [card_id(4)]},
        phase=Phase.DRAGON_GIFT,
        current_player=0,
        trick_cards=trick_cards,
        pending_dragon_winner=0,
    )

    actions = round_.legal_actions()
    assert {action.recipient for action in actions} == {1, 3}

    round_ = round_.apply(DragonGiftAction(1))

    assert round_.captured[1][-2:] == tuple(trick_cards)
    assert round_.pending_dragon_winner is None
    assert round_.phase is Phase.PLAY


def test_dragon_trick_can_be_bombed_before_recipient_selection() -> None:
    dragon = combo_of_kind((card_id(special=Special.DRAGON),), CombinationKind.SINGLE)
    bomb = combo_of_kind(normal_ids(3), CombinationKind.FOUR_BOMB)
    round_ = make_round(
        {
            0: [card_id(special=Special.DRAGON)],
            1: list(bomb.card_ids),
            2: [card_id(4)],
            3: [card_id(5)],
        },
        config=_out_of_turn_bomb_config(),
        current_player=0,
    )

    round_ = round_.apply(PlayAction(dragon))
    assert round_.phase is Phase.BOMB_WINDOW
    assert round_.current_player == 1

    round_ = round_.apply(BombAction(bomb))

    assert round_.phase is not Phase.DRAGON_GIFT
    assert round_.pending_dragon_winner is None
    assert card_id(special=Special.DRAGON) in round_.captured[1]


def test_bombs_use_normal_turn_order_by_default() -> None:
    first_bomb = combo_of_kind(normal_ids(5), CombinationKind.FOUR_BOMB)
    higher_bomb = combo_of_kind(normal_ids(6), CombinationKind.FOUR_BOMB)
    round_ = make_round(
        {
            0: [*first_bomb.card_ids, card_id(9)],
            1: [*higher_bomb.card_ids, card_id(10)],
            2: [card_id(2)],
            3: [card_id(3)],
        },
        current_player=0,
    )

    round_ = round_.apply(PlayAction(first_bomb))

    assert round_.phase is Phase.PLAY
    assert round_.current_player == 1
    assert round_.bomb_queue == ()
    assert round_.bomb_resume_player is None
    assert round_.bomb_close_behavior is None
    actions = round_.legal_actions()
    assert PassAction() in actions
    assert PlayAction(higher_bomb) in actions
    assert not any(isinstance(action, BombAction) for action in actions)


def test_turn_only_bomb_trick_resolves_through_normal_passes() -> None:
    bomb = combo_of_kind(normal_ids(5), CombinationKind.FOUR_BOMB)
    round_ = make_round(
        {
            0: [*bomb.card_ids, card_id(9)],
            1: [card_id(10)],
            2: [card_id(2)],
            3: [card_id(3)],
        },
        current_player=0,
    )

    round_ = round_.apply(PlayAction(bomb))
    for player in (1, 2, 3):
        assert round_.current_player == player
        round_ = round_.apply(PassAction())
        assert round_.phase is not Phase.BOMB_WINDOW

    assert round_.current_player == 0
    assert round_.top_combination is None
    assert set(bomb.card_ids).issubset(round_.captured[0])


def test_player_can_go_out_by_playing_a_bomb() -> None:
    top = combo_of_kind((card_id(6),), CombinationKind.SINGLE)
    bomb = combo_of_kind(normal_ids(7), CombinationKind.FOUR_BOMB)
    round_ = make_round(
        {
            0: [],
            1: list(bomb.card_ids),
            2: [card_id(2)],
            3: [card_id(3)],
        },
        current_player=1,
        top_combination=top,
        trick_cards=top.card_ids,
        last_play_player=0,
    )

    round_ = round_.apply(PlayAction(bomb))

    assert 1 in round_.finish_order


def test_rule_config_removes_disabled_actions_and_phases() -> None:
    config = TichuRoundConfig(
        grand_tichu=False,
        small_tichu=False,
        card_exchange=False,
        bombs=False,
        wishes=False,
        special_cards=True,
    )
    mahjong = combo_of_kind((card_id(special=Special.MAHJONG),), CombinationKind.SINGLE)
    round_ = make_round(
        {
            0: [card_id(special=Special.MAHJONG)],
            1: list(normal_ids(5)),
            2: [card_id(2)],
            3: [card_id(3)],
        },
        config=config,
        current_player=0,
    )

    actions = round_.legal_actions()
    assert not any(action.__class__.__name__ == "DeclareSmallTichu" for action in actions)
    assert not any(isinstance(action, BombAction) for action in actions)
    assert any(isinstance(action, PlayAction) and action.wish_rank is None for action in actions)
    assert not any(isinstance(action, PlayAction) and action.wish_rank for action in actions)

    round_ = round_.apply(PlayAction(mahjong, None))

    assert round_.wish_rank is None
    assert round_.phase not in {Phase.GRAND_TICHU, Phase.EXCHANGE, Phase.BOMB_WINDOW}
