from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Iterable


class Suit(str, Enum):
    JADE = "jade"
    SWORD = "sword"
    PAGODA = "pagoda"
    STAR = "star"


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class Special(str, Enum):
    MAHJONG = "mahjong"
    DOG = "dog"
    PHOENIX = "phoenix"
    DRAGON = "dragon"


@dataclass(frozen=True, order=True)
class Card:
    """One physical card from a Tichu deck.

    ``card_id`` is stable and unique, which makes replay logs and bit-mask
    encodings straightforward later.
    """

    card_id: int
    suit: Suit | None = None
    rank: int | None = None
    special: Special | None = None

    def __post_init__(self) -> None:
        normal = self.rank is not None and self.suit is not None and self.special is None
        special = self.special is not None and self.rank is None and self.suit is None
        if not (normal or special):
            raise ValueError("A card must be either a normal suited card or one special card")
        if self.rank is not None and self.rank not in range(2, 15):
            raise ValueError(f"Invalid normal rank: {self.rank}")

    @property
    def is_normal(self) -> bool:
        return self.special is None

    @property
    def label(self) -> str:
        if self.special is not None:
            return self.special.value.upper()
        rank_labels = {11: "J", 12: "Q", 13: "K", 14: "A"}
        return f"{rank_labels.get(self.rank, self.rank)}-{self.suit.value}"

    @property
    def points(self) -> int:
        if self.special is Special.DRAGON:
            return 25
        if self.special is Special.PHOENIX:
            return -25
        if self.rank == 5:
            return 5
        if self.rank in (10, 13):
            return 10
        return 0


def build_deck() -> tuple[Card, ...]:
    cards: list[Card] = []
    card_id = 0
    for suit in Suit:
        for rank in range(2, 15):
            cards.append(Card(card_id=card_id, suit=suit, rank=rank))
            card_id += 1
    for special in (Special.MAHJONG, Special.DOG, Special.PHOENIX, Special.DRAGON):
        cards.append(Card(card_id=card_id, special=special))
        card_id += 1
    return tuple(cards)


DECK: tuple[Card, ...] = build_deck()
CARD_BY_ID: dict[int, Card] = {card.card_id: card for card in DECK}


def cards_from_ids(card_ids: Iterable[int]) -> tuple[Card, ...]:
    try:
        return tuple(CARD_BY_ID[card_id] for card_id in card_ids)
    except KeyError as exc:
        raise ValueError(f"Unknown card id: {exc.args[0]}") from exc


def deck_points(cards: Iterable[Card] = DECK) -> int:
    return sum(card.points for card in cards)
