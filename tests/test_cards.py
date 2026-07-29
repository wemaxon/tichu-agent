from __future__ import annotations

from tichu.cards import CARD_BY_ID, CARD_CLASS_NAMES, DECK, Special, Suit, deck_points


EXPECTED_CARD_CLASS_NAMES = (
    "10-sword",
    "2-sword",
    "3-sword",
    "4-sword",
    "5-sword",
    "6-sword",
    "7-sword",
    "8-sword",
    "9-sword",
    "ace-sword",
    "jack-sword",
    "king-sword",
    "queen-sword",
    "10-pagoda",
    "2-pagoda",
    "3-pagoda",
    "4-pagoda",
    "5-pagoda",
    "6-pagoda",
    "7-pagoda",
    "8-pagoda",
    "9-pagoda",
    "ace-pagoda",
    "jack-pagoda",
    "king-pagoda",
    "queen-pagoda",
    "dog",
    "dragon",
    "10-jade",
    "2-jade",
    "3-jade",
    "4-jade",
    "5-jade",
    "6-jade",
    "7-jade",
    "8-jade",
    "9-jade",
    "ace-jade",
    "jack-jade",
    "king-jade",
    "queen-jade",
    "mahjong",
    "phoenix",
    "10-star",
    "2-star",
    "3-star",
    "4-star",
    "5-star",
    "6-star",
    "7-star",
    "8-star",
    "9-star",
    "ace-star",
    "jack-star",
    "king-star",
    "queen-star",
)


def test_card_class_names_match_canonical_identity_mapping() -> None:
    assert CARD_CLASS_NAMES == EXPECTED_CARD_CLASS_NAMES
    assert tuple(card.card_id for card in DECK) == tuple(range(56))
    assert tuple(CARD_BY_ID[card_id].label for card_id in range(56)) == EXPECTED_CARD_CLASS_NAMES


def test_card_class_ids_have_expected_semantics() -> None:
    assert CARD_BY_ID[0].rank == 10
    assert CARD_BY_ID[0].suit is Suit.SWORD
    assert CARD_BY_ID[9].rank == 14
    assert CARD_BY_ID[9].suit is Suit.SWORD
    assert CARD_BY_ID[11].rank == 13
    assert CARD_BY_ID[11].suit is Suit.SWORD
    assert CARD_BY_ID[12].rank == 12
    assert CARD_BY_ID[12].suit is Suit.SWORD
    assert CARD_BY_ID[26].special is Special.DOG
    assert CARD_BY_ID[27].special is Special.DRAGON
    assert CARD_BY_ID[41].special is Special.MAHJONG
    assert CARD_BY_ID[42].special is Special.PHOENIX


def test_canonical_deck_invariants() -> None:
    assert len(DECK) == 56
    assert len(CARD_BY_ID) == 56
    assert set(CARD_BY_ID) == set(range(56))
    assert deck_points() == 100
