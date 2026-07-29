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


RANK_ORDER: tuple[Rank, ...] = (
    Rank.TEN,
    Rank.TWO,
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.ACE,
    Rank.JACK,
    Rank.KING,
    Rank.QUEEN,
)

CARD_ID_LAYOUT: tuple[Suit | Special, ...] = (
    Suit.SWORD,
    Suit.PAGODA,
    Special.DOG,
    Special.DRAGON,
    Suit.JADE,
    Special.MAHJONG,
    Special.PHOENIX,
    Suit.STAR,
)

_RANK_LABELS: dict[int, str] = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "jack",
    12: "queen",
    13: "king",
    14: "ace",
}


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
    def normal_rank(self) -> int:
        if self.rank is None:
            raise ValueError("Special cards do not have a normal rank")
        return self.rank

    @property
    def normal_suit(self) -> Suit:
        if self.suit is None:
            raise ValueError("Special cards do not have a suit")
        return self.suit

    @property
    def label(self) -> str:
        if self.special is not None:
            return self.special.value
        return f"{_RANK_LABELS[self.normal_rank]}-{self.normal_suit.value}"

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
    for item in CARD_ID_LAYOUT:
        if isinstance(item, Suit):
            for rank in RANK_ORDER:
                cards.append(Card(card_id=len(cards), suit=item, rank=rank))
        else:
            cards.append(Card(card_id=len(cards), special=item))
    return tuple(cards)


DECK: tuple[Card, ...] = build_deck()
CARD_BY_ID: dict[int, Card] = {card.card_id: card for card in DECK}
CARD_CLASS_NAMES: tuple[str, ...] = tuple(card.label for card in DECK)


def cards_from_ids(card_ids: Iterable[int]) -> tuple[Card, ...]:
    try:
        return tuple(CARD_BY_ID[card_id] for card_id in card_ids)
    except KeyError as exc:
        raise ValueError(f"Unknown card id: {exc.args[0]}") from exc


def deck_points(cards: Iterable[Card] = DECK) -> int:
    return sum(card.points for card in cards)
