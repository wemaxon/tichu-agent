from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations as choose

import pytest

from tichu.cards import CARD_BY_ID, DECK, Special, Suit
from tichu.combinations import (
    Combination,
    CombinationKind,
    classify_all,
    generate_bombs,
    generate_combinations,
    generate_combinations_for_context,
    resolve_for_context,
)

from tests.helpers import card_id, classify_ids, combo_of_kind, combos_of_kind, normal_ids


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


def test_bomb_generator_matches_exhaustive_bomb_filter() -> None:
    hand = (
        *normal_ids(9),
        *(card_id(rank, Suit.JADE) for rank in range(3, 9)),
        card_id(special=Special.PHOENIX),
    )

    expected = tuple(
        combo
        for combo in _exhaustive_combinations(tuple(sorted(hand)), True)
        if combo.is_bomb
    )

    assert generate_bombs(tuple(sorted(hand))) == expected


@pytest.mark.parametrize("allow_bombs", (False, True))
def test_indexed_generator_matches_exhaustive_classifier(
    allow_bombs: bool,
) -> None:
    rng = random.Random(20260731)
    phoenix = card_id(special=Special.PHOENIX)
    mahjong = card_id(special=Special.MAHJONG)

    for size in range(1, 15):
        hands = [
            tuple(sorted(card.card_id for card in rng.sample(DECK, size))),
            _sample_hand_with_required_cards(
                rng,
                size,
                required=(phoenix,),
            ),
        ]
        if size >= 2:
            hands.append(
                _sample_hand_with_required_cards(
                    rng,
                    size,
                    required=(mahjong, phoenix),
                )
            )

        for hand in hands:
            assert generate_combinations(
                hand,
                allow_bombs,
            ) == _exhaustive_combinations(hand, allow_bombs)


@pytest.mark.parametrize(
    "top",
    [
        combo_of_kind((card_id(5, Suit.PAGODA),), CombinationKind.SINGLE),
        combo_of_kind((card_id(2, Suit.PAGODA), card_id(2, Suit.STAR)), CombinationKind.PAIR),
        combo_of_kind(
            (card_id(2, Suit.PAGODA), card_id(2, Suit.STAR), card_id(2, Suit.SWORD)),
            CombinationKind.TRIPLE,
        ),
        combo_of_kind(
            (
                card_id(2, Suit.PAGODA),
                card_id(2, Suit.STAR),
                card_id(2, Suit.SWORD),
                card_id(3, Suit.PAGODA),
                card_id(3, Suit.STAR),
            ),
            CombinationKind.FULL_HOUSE,
        ),
        combo_of_kind(
            (
                card_id(2, Suit.PAGODA),
                card_id(2, Suit.STAR),
                card_id(3, Suit.PAGODA),
                card_id(3, Suit.STAR),
            ),
            CombinationKind.PAIR_SEQUENCE,
        ),
        combo_of_kind(
            (
                card_id(2, Suit.PAGODA),
                card_id(3, Suit.STAR),
                card_id(4, Suit.SWORD),
                card_id(5, Suit.PAGODA),
                card_id(6, Suit.STAR),
            ),
            CombinationKind.STRAIGHT,
        ),
        combo_of_kind(
            tuple(card_id(rank, Suit.PAGODA) for rank in range(2, 7)),
            CombinationKind.STRAIGHT_FLUSH_BOMB,
        ),
        combo_of_kind(
            tuple(card_id(rank, Suit.PAGODA) for rank in range(3, 8)),
            CombinationKind.STRAIGHT_FLUSH_BOMB,
        ),
        combo_of_kind((card_id(special=Special.DRAGON),), CombinationKind.SINGLE),
    ],
)
def test_contextual_generator_matches_exhaustive_filter(top: Combination) -> None:
    hand = (
        card_id(special=Special.MAHJONG),
        card_id(special=Special.PHOENIX),
        card_id(special=Special.DOG),
        card_id(special=Special.DRAGON),
        card_id(2, Suit.JADE),
        card_id(2, Suit.SWORD),
        card_id(3, Suit.JADE),
        card_id(3, Suit.SWORD),
        card_id(4, Suit.JADE),
        card_id(5, Suit.JADE),
        card_id(6, Suit.JADE),
        card_id(7, Suit.JADE),
        card_id(8, Suit.JADE),
        card_id(9, Suit.JADE),
    )

    assert generate_combinations_for_context(hand, top, allow_bombs=True) == _exhaustive_context(
        hand,
        top,
        allow_bombs=True,
    )


def test_contextual_generator_treats_disabled_bombs_as_regular_straights() -> None:
    hand = tuple(card_id(rank, Suit.JADE) for rank in range(5, 10))
    top = combo_of_kind(
        tuple(card_id(rank, Suit.PAGODA) for rank in range(4, 9)),
        CombinationKind.STRAIGHT,
        allow_bombs=False,
    )

    generated = generate_combinations_for_context(hand, top, allow_bombs=False)

    assert generated == _exhaustive_context(hand, top, allow_bombs=False)
    assert any(combo.kind is CombinationKind.STRAIGHT for combo in generated)
    assert not any(combo.is_bomb for combo in generated)


def test_contextual_bombs_only_generator_matches_exhaustive_filter() -> None:
    hand = (
        *normal_ids(9),
        *(card_id(rank, Suit.JADE) for rank in range(3, 9)),
    )
    top = combo_of_kind(normal_ids(8), CombinationKind.FOUR_BOMB)

    generated = generate_combinations_for_context(
        tuple(sorted(hand)),
        top,
        allow_bombs=True,
        bombs_only=True,
    )

    assert generated == _exhaustive_context(
        hand,
        top,
        allow_bombs=True,
        bombs_only=True,
    )
    assert all(combo.is_bomb for combo in generated)


def _exhaustive_context(
    hand_ids: tuple[int, ...],
    top: Combination | None,
    *,
    allow_bombs: bool,
    bombs_only: bool = False,
) -> tuple[Combination, ...]:
    results: set[Combination] = set()
    for combo in _exhaustive_combinations(tuple(sorted(hand_ids)), allow_bombs):
        if bombs_only and not combo.is_bomb:
            continue
        if top is None:
            results.add(combo)
            continue
        try:
            resolved = resolve_for_context(combo, top)
        except ValueError:
            continue
        if resolved.beats(top):
            results.add(resolved)
    return tuple(sorted(results, key=Combination.sort_key))


@lru_cache(maxsize=None)
def _exhaustive_combinations(
    hand_ids: tuple[int, ...],
    allow_bombs: bool,
) -> tuple[Combination, ...]:
    cards = tuple(CARD_BY_ID[card_id_] for card_id_ in hand_ids)
    results: set[Combination] = set()
    for size in range(1, len(cards) + 1):
        for subset in choose(cards, size):
            results.update(
                classify_all(
                    subset,
                    allow_bombs=allow_bombs,
                )
            )
    return tuple(sorted(results, key=Combination.sort_key))


def _sample_hand_with_required_cards(
    rng: random.Random,
    size: int,
    *,
    required: tuple[int, ...],
) -> tuple[int, ...]:
    required_set = set(required)
    available = [
        card.card_id
        for card in DECK
        if card.card_id not in required_set
    ]
    return tuple(
        sorted(
            (*required, *rng.sample(available, size - len(required)))
        )
    )
