from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from itertools import combinations as choose
from collections import Counter

from .cards import CARD_BY_ID, Card, Special


class CombinationKind(str, Enum):
    DOG = "dog"
    SINGLE = "single"
    PAIR = "pair"
    TRIPLE = "triple"
    FULL_HOUSE = "full_house"
    PAIR_SEQUENCE = "pair_sequence"
    STRAIGHT = "straight"
    FOUR_BOMB = "four_bomb"
    STRAIGHT_FLUSH_BOMB = "straight_flush_bomb"


_KIND_ORDER = {
    CombinationKind.DOG: 0,
    CombinationKind.SINGLE: 1,
    CombinationKind.PAIR: 2,
    CombinationKind.TRIPLE: 3,
    CombinationKind.FULL_HOUSE: 4,
    CombinationKind.PAIR_SEQUENCE: 5,
    CombinationKind.STRAIGHT: 6,
    CombinationKind.FOUR_BOMB: 7,
    CombinationKind.STRAIGHT_FLUSH_BOMB: 8,
}


@dataclass(frozen=True)
class Combination:
    card_ids: tuple[int, ...]
    kind: CombinationKind
    primary_rank: float
    length: int
    phoenix_as: int | None = None

    @property
    def is_bomb(self) -> bool:
        return self.kind in {
            CombinationKind.FOUR_BOMB,
            CombinationKind.STRAIGHT_FLUSH_BOMB,
        }

    @property
    def is_dog(self) -> bool:
        return self.kind is CombinationKind.DOG

    @property
    def cards(self) -> tuple[Card, ...]:
        return tuple(CARD_BY_ID[card_id] for card_id in self.card_ids)

    @property
    def bomb_key(self) -> tuple[int, float]:
        if not self.is_bomb:
            raise ValueError("Only bombs have bomb strength")
        # All straight-flush bombs have >= 5 cards, so length first also makes
        # every straight-flush bomb stronger than every four-of-a-kind bomb.
        return self.length, self.primary_rank

    def contains_special(self, special: Special) -> bool:
        return any(card.special is special for card in self.cards)

    def contains_normal_rank(self, rank: int) -> bool:
        return any(card.is_normal and card.normal_rank == rank for card in self.cards)

    def beats(self, other: Combination) -> bool:
        if self.is_dog or other.is_dog:
            return False
        if self.is_bomb:
            return not other.is_bomb or self.bomb_key > other.bomb_key
        if other.is_bomb:
            return False
        if self.kind is not other.kind or self.length != other.length:
            return False
        return self.primary_rank > other.primary_rank

    def sort_key(self) -> tuple:
        return (
            _KIND_ORDER[self.kind],
            self.length,
            self.primary_rank,
            -1 if self.phoenix_as is None else self.phoenix_as,
            self.card_ids,
        )


def _combo(
    cards: tuple[Card, ...],
    kind: CombinationKind,
    primary_rank: float,
    phoenix_as: int | None = None,
) -> Combination:
    return Combination(
        card_ids=tuple(sorted(card.card_id for card in cards)),
        kind=kind,
        primary_rank=primary_rank,
        length=len(cards),
        phoenix_as=phoenix_as,
    )


def classify_all(cards: tuple[Card, ...], *, allow_bombs: bool = True) -> tuple[Combination, ...]:
    """Return every legal interpretation of exactly ``cards``.

    Multiple interpretations matter when the Phoenix can fill different ranks,
    for example ``3,3,4,4,Phoenix`` can be a full house whose triple is either
    threes or fours.
    """

    cards = tuple(sorted(cards, key=lambda card: card.card_id))
    n = len(cards)
    if n == 0:
        return ()

    phoenixes = [card for card in cards if card.special is Special.PHOENIX]
    if len(phoenixes) > 1:
        return ()
    phoenix = phoenixes[0] if phoenixes else None
    other_specials = [
        card
        for card in cards
        if card.special is not None and card.special is not Special.PHOENIX
    ]
    normals = [card for card in cards if card.is_normal]
    results: set[Combination] = set()

    if n == 1:
        card = cards[0]
        if card.special is Special.DOG:
            results.add(_combo(cards, CombinationKind.DOG, 0))
        elif card.special is Special.MAHJONG:
            results.add(_combo(cards, CombinationKind.SINGLE, 1))
        elif card.special is Special.PHOENIX:
            # Contextual value is resolved when it is played; 1.5 is its lead value.
            results.add(_combo(cards, CombinationKind.SINGLE, 1.5))
        elif card.special is Special.DRAGON:
            results.add(_combo(cards, CombinationKind.SINGLE, 15))
        else:
            results.add(_combo(cards, CombinationKind.SINGLE, float(card.normal_rank)))
        return tuple(sorted(results, key=Combination.sort_key))

    # Pairs and triples use only normal cards, with the Phoenix substituting for
    # one normal card of the same rank.
    if not other_specials:
        normal_counts = Counter(card.normal_rank for card in normals)
        if n == 2:
            if phoenix is None and len(normal_counts) == 1:
                rank = next(iter(normal_counts))
                results.add(_combo(cards, CombinationKind.PAIR, float(rank)))
            elif phoenix is not None and len(normals) == 1:
                rank = normals[0].normal_rank
                results.add(_combo(cards, CombinationKind.PAIR, float(rank), rank))

        if n == 3:
            if phoenix is None and list(normal_counts.values()) == [3]:
                rank = next(iter(normal_counts))
                results.add(_combo(cards, CombinationKind.TRIPLE, float(rank)))
            elif phoenix is not None and len(normal_counts) == 1 and len(normals) == 2:
                rank = next(iter(normal_counts))
                results.add(_combo(cards, CombinationKind.TRIPLE, float(rank), rank))

        # Four equal normal cards are a bomb.
        if allow_bombs and n == 4 and phoenix is None and list(normal_counts.values()) == [4]:
            rank = next(iter(normal_counts))
            results.add(_combo(cards, CombinationKind.FOUR_BOMB, float(rank)))

        # Full house. 3333+Phoenix is deliberately rejected, matching the FAQ.
        if n == 5:
            if phoenix is None and sorted(normal_counts.values()) == [2, 3]:
                triple_rank = next(rank for rank, count in normal_counts.items() if count == 3)
                results.add(_combo(cards, CombinationKind.FULL_HOUSE, float(triple_rank)))
            elif phoenix is not None and len(normals) == 4:
                counts = sorted(normal_counts.values())
                if counts == [1, 3]:
                    triple_rank = next(rank for rank, count in normal_counts.items() if count == 3)
                    pair_rank = next(rank for rank, count in normal_counts.items() if count == 1)
                    results.add(
                        _combo(
                            cards,
                            CombinationKind.FULL_HOUSE,
                            float(triple_rank),
                            pair_rank,
                        )
                    )
                elif counts == [2, 2]:
                    for triple_rank in sorted(normal_counts):
                        results.add(
                            _combo(
                                cards,
                                CombinationKind.FULL_HOUSE,
                                float(triple_rank),
                                triple_rank,
                            )
                        )

        # Consecutive pairs (sometimes called a tractor).
        if n >= 4 and n % 2 == 0:
            pair_count = n // 2
            if all(count <= 2 for count in normal_counts.values()):
                ranks = sorted(normal_counts)
                if phoenix is None:
                    if (
                        len(ranks) == pair_count
                        and all(normal_counts[rank] == 2 for rank in ranks)
                        and ranks == list(range(ranks[0], ranks[0] + pair_count))
                    ):
                        results.add(
                            _combo(cards, CombinationKind.PAIR_SEQUENCE, float(ranks[-1]))
                        )
                else:
                    singleton_ranks = [rank for rank, count in normal_counts.items() if count == 1]
                    if (
                        len(ranks) == pair_count
                        and len(singleton_ranks) == 1
                        and all(count in (1, 2) for count in normal_counts.values())
                        and ranks == list(range(ranks[0], ranks[0] + pair_count))
                    ):
                        missing_partner = singleton_ranks[0]
                        results.add(
                            _combo(
                                cards,
                                CombinationKind.PAIR_SEQUENCE,
                                float(ranks[-1]),
                                missing_partner,
                            )
                        )

    # Straights can contain the Mahjong as rank 1 and/or Phoenix as one normal
    # rank. Dog and Dragon can never be part of a straight.
    if n >= 5 and all(
        card.special in (None, Special.MAHJONG, Special.PHOENIX) for card in cards
    ):
        explicit_ranks: list[int] = []
        duplicate = False
        for card in cards:
            if card.is_normal:
                rank = card.normal_rank
            elif card.special is Special.MAHJONG:
                rank = 1
            else:
                continue
            if rank in explicit_ranks:
                duplicate = True
                break
            explicit_ranks.append(rank)

        if not duplicate:
            explicit = set(explicit_ranks)
            if phoenix is None:
                ordered = sorted(explicit)
                if len(ordered) == n and ordered == list(range(ordered[0], ordered[0] + n)):
                    same_suit = len(normals) == n and len({card.normal_suit for card in normals}) == 1
                    kind = (
                        CombinationKind.STRAIGHT_FLUSH_BOMB
                        if allow_bombs and same_suit
                        else CombinationKind.STRAIGHT
                    )
                    results.add(_combo(cards, kind, float(ordered[-1])))
            else:
                for start in range(1, 15 - n + 1):
                    interval = set(range(start, start + n))
                    missing = interval - explicit
                    if len(missing) != 1 or not explicit.issubset(interval):
                        continue
                    missing_rank = next(iter(missing))
                    # Phoenix may impersonate only a normal 2-A card, not Mahjong.
                    if missing_rank < 2:
                        continue
                    results.add(
                        _combo(
                            cards,
                            CombinationKind.STRAIGHT,
                            float(start + n - 1),
                            missing_rank,
                        )
                    )

    return tuple(sorted(results, key=Combination.sort_key))


@lru_cache(maxsize=8192)
def generate_combinations(
    hand_ids: tuple[int, ...],
    allow_bombs: bool = True,
) -> tuple[Combination, ...]:
    """Generate all legal combinations contained in a hand.

    A 14-card hand has only 16,383 non-empty subsets. Exhaustive subset
    classification keeps the rules implementation easy to audit and is fast
    enough for a correctness-first simulator. A future RL rollout engine can
    replace this with indexed generators without changing the public API.
    """

    hand_ids = tuple(sorted(hand_ids))
    cards = tuple(CARD_BY_ID[card_id] for card_id in hand_ids)
    results: set[Combination] = set()
    for size in range(1, len(cards) + 1):
        for subset in choose(cards, size):
            results.update(classify_all(subset, allow_bombs=allow_bombs))
    return tuple(sorted(results, key=Combination.sort_key))


def resolve_for_context(combo: Combination, top: Combination | None) -> Combination:
    """Resolve the contextual value of a Phoenix played as a single."""

    if not (
        combo.kind is CombinationKind.SINGLE
        and combo.contains_special(Special.PHOENIX)
    ):
        return combo
    if top is None:
        return replace(combo, primary_rank=1.5)
    if top.kind is not CombinationKind.SINGLE or top.is_bomb:
        raise ValueError("A single Phoenix can only answer another single")
    if top.contains_special(Special.DRAGON):
        raise ValueError("The Phoenix cannot beat the Dragon")
    return replace(combo, primary_rank=min(14.5, top.primary_rank + 0.5))
