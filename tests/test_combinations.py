from __future__ import annotations

import pytest

from tichu.cards import Special, Suit
from tichu.combinations import CombinationKind, resolve_for_context

from .helpers import card_id, classify_ids, combo_of_kind, combos_of_kind, normal_ids


pytestmark = pytest.mark.rules


def test_phoenix_forms_pairs_triples_full_houses_and_straights() -> None:
    phoenix = card_id(special=Special.PHOENIX)

    pair = combo_of_kind(
        (card_id(5), phoenix),
        CombinationKind.PAIR,
        primary_rank=5,
        phoenix_as=5,
    )
    assert pair.length == 2

    triple = combo_of_kind(
        (card_id(7, Suit.JADE), card_id(7, Suit.SWORD), phoenix),
        CombinationKind.TRIPLE,
        primary_rank=7,
        phoenix_as=7,
    )
    assert triple.length == 3

    full_houses = combos_of_kind(
        (
            card_id(3, Suit.JADE),
            card_id(3, Suit.SWORD),
            card_id(4, Suit.JADE),
            card_id(4, Suit.SWORD),
            phoenix,
        ),
        CombinationKind.FULL_HOUSE,
    )
    assert {combo.primary_rank for combo in full_houses} == {3, 4}
    assert {combo.phoenix_as for combo in full_houses} == {3, 4}

    straight = combo_of_kind(
        (
            card_id(3),
            card_id(4),
            card_id(6),
            card_id(7),
            phoenix,
        ),
        CombinationKind.STRAIGHT,
        primary_rank=7,
        phoenix_as=5,
    )
    assert straight.length == 5


def test_phoenix_does_not_form_bombs() -> None:
    phoenix = card_id(special=Special.PHOENIX)
    three_sevens_and_phoenix = (*normal_ids(7)[:3], phoenix)
    assert not any(combo.is_bomb for combo in classify_ids(*three_sevens_and_phoenix))

    straight_with_phoenix = (
        card_id(3, Suit.JADE),
        card_id(4, Suit.JADE),
        card_id(6, Suit.JADE),
        card_id(7, Suit.JADE),
        phoenix,
    )
    assert not any(combo.is_bomb for combo in classify_ids(*straight_with_phoenix))
    assert combos_of_kind(straight_with_phoenix, CombinationKind.STRAIGHT)


def test_phoenix_single_lead_value() -> None:
    phoenix_single = combo_of_kind(
        (card_id(special=Special.PHOENIX),),
        CombinationKind.SINGLE,
    )

    assert resolve_for_context(phoenix_single, None).primary_rank == 1.5


def test_phoenix_single_after_mahjong() -> None:
    phoenix_single = combo_of_kind(
        (card_id(special=Special.PHOENIX),),
        CombinationKind.SINGLE,
    )
    mahjong = combo_of_kind(
        (card_id(special=Special.MAHJONG),),
        CombinationKind.SINGLE,
    )

    assert resolve_for_context(phoenix_single, mahjong).primary_rank == 1.5


@pytest.mark.parametrize("rank", range(2, 15))
def test_phoenix_single_resolves_after_normal_rank(rank: int) -> None:
    phoenix_single = combo_of_kind(
        (card_id(special=Special.PHOENIX),),
        CombinationKind.SINGLE,
    )
    top = combo_of_kind((card_id(rank),), CombinationKind.SINGLE)

    resolved = resolve_for_context(phoenix_single, top)

    assert resolved.primary_rank == min(14.5, rank + 0.5)


def test_phoenix_single_cannot_beat_dragon() -> None:
    phoenix_single = combo_of_kind(
        (card_id(special=Special.PHOENIX),),
        CombinationKind.SINGLE,
    )
    dragon = combo_of_kind(
        (card_id(special=Special.DRAGON),),
        CombinationKind.SINGLE,
    )

    with pytest.raises(ValueError):
        resolve_for_context(phoenix_single, dragon)


def test_four_of_a_kind_and_straight_flush_bombs() -> None:
    four_bomb = combo_of_kind(normal_ids(9), CombinationKind.FOUR_BOMB, primary_rank=9)
    assert four_bomb.bomb_key == (4, 9)

    five_card_bomb = combo_of_kind(
        [card_id(rank, Suit.JADE) for rank in range(3, 8)],
        CombinationKind.STRAIGHT_FLUSH_BOMB,
        primary_rank=7,
    )
    six_card_bomb = combo_of_kind(
        [card_id(rank, Suit.SWORD) for rank in range(3, 9)],
        CombinationKind.STRAIGHT_FLUSH_BOMB,
        primary_rank=8,
    )
    assert five_card_bomb.bomb_key == (5, 7)
    assert six_card_bomb.bomb_key == (6, 8)
    assert six_card_bomb.beats(five_card_bomb)


def test_bomb_versus_bomb_ranking() -> None:
    king_four_bomb = combo_of_kind(normal_ids(13), CombinationKind.FOUR_BOMB)
    ace_four_bomb = combo_of_kind(normal_ids(14), CombinationKind.FOUR_BOMB)
    straight_flush_five = combo_of_kind(
        [card_id(rank, Suit.PAGODA) for rank in range(2, 7)],
        CombinationKind.STRAIGHT_FLUSH_BOMB,
    )
    straight_flush_six_low = combo_of_kind(
        [card_id(rank, Suit.STAR) for rank in range(2, 8)],
        CombinationKind.STRAIGHT_FLUSH_BOMB,
    )

    assert ace_four_bomb.beats(king_four_bomb)
    assert straight_flush_five.beats(ace_four_bomb)
    assert straight_flush_six_low.beats(straight_flush_five)


def test_disabled_bombs_are_generated_as_non_bomb_straights_only() -> None:
    four_kind = normal_ids(6)
    assert not any(combo.is_bomb for combo in classify_ids(*four_kind, allow_bombs=False))

    straight_flush = [card_id(rank, Suit.JADE) for rank in range(5, 10)]
    combos = classify_ids(*straight_flush, allow_bombs=False)
    assert not any(combo.is_bomb for combo in combos)
    assert any(combo.kind is CombinationKind.STRAIGHT for combo in combos)
