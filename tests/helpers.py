from __future__ import annotations

from collections.abc import Iterable

from tichu.actions import ExchangeAction
from tichu.cards import CARD_BY_ID, DECK, Special, Suit
from tichu.combinations import Combination, CombinationKind, classify_all
from tichu.config import TichuRoundConfig
from tichu.round import BombCloseBehavior, Phase, TichuRound


def card_id(rank: int | None = None, suit: Suit = Suit.JADE, special: Special | None = None) -> int:
    for card in DECK:
        if special is not None and card.special is special:
            return card.card_id
        if special is None and card.special is None and card.rank == rank and card.suit is suit:
            return card.card_id
    raise LookupError(f"No card for rank={rank!r}, suit={suit!r}, special={special!r}")


def normal_ids(rank: int) -> tuple[int, ...]:
    return tuple(card_id(rank, suit) for suit in Suit)


def classify_ids(*ids: int, allow_bombs: bool = True) -> tuple[Combination, ...]:
    return classify_all(tuple(CARD_BY_ID[card_id_] for card_id_ in ids), allow_bombs=allow_bombs)


def combos_of_kind(
    ids: Iterable[int],
    kind: CombinationKind,
    *,
    allow_bombs: bool = True,
) -> list[Combination]:
    return [
        combo
        for combo in classify_ids(*tuple(ids), allow_bombs=allow_bombs)
        if combo.kind is kind
    ]


def combo_of_kind(
    ids: Iterable[int],
    kind: CombinationKind,
    *,
    primary_rank: float | None = None,
    phoenix_as: int | None | object = Ellipsis,
    allow_bombs: bool = True,
) -> Combination:
    matches = combos_of_kind(ids, kind, allow_bombs=allow_bombs)
    if primary_rank is not None:
        matches = [combo for combo in matches if combo.primary_rank == primary_rank]
    if phoenix_as is not Ellipsis:
        matches = [combo for combo in matches if combo.phoenix_as == phoenix_as]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {kind.value} from {tuple(ids)}, got {matches}")
    return matches[0]


def make_round(
    hands: dict[int, Iterable[int]],
    *,
    config: TichuRoundConfig | None = None,
    dealer: int = 0,
    phase: Phase = Phase.PLAY,
    current_player: int | None = 0,
    captured: dict[int, Iterable[int]] | None = None,
    discarded: Iterable[int] = (),
    deck_remaining: Iterable[int] = (),
    pending_exchanges: dict[int, ExchangeAction] | None = None,
    tichu_calls: dict[int, int] | None = None,
    has_played_card: Iterable[bool] | None = None,
    wish_rank: int | None = None,
    top_combination: Combination | None = None,
    trick_cards: Iterable[int] = (),
    last_play_player: int | None = None,
    consecutive_passes: int = 0,
    finish_order: Iterable[int] = (),
    bomb_queue: Iterable[int] = (),
    bomb_resume_player: int | None = None,
    bomb_close_behavior: BombCloseBehavior | None = None,
    pending_dragon_winner: int | None = None,
    finalize_after_dragon: bool = False,
    round_scores: tuple[int, int] | None = None,
) -> TichuRound:
    config = TichuRoundConfig() if config is None else config
    round_hands = {player: list(hands.get(player, ())) for player in range(4)}
    round_captured = {
        player: list((captured or {}).get(player, ()))
        for player in range(4)
    }
    round_discarded = list(discarded)
    round_deck_remaining = list(deck_remaining)
    round_trick_cards = list(trick_cards)

    expected = {
        card.card_id
        for card in DECK
        if config.special_cards or card.special is None
    }
    used = []
    for zone in round_hands.values():
        used.extend(zone)
    for zone in round_captured.values():
        used.extend(zone)
    used.extend(round_discarded)
    used.extend(round_deck_remaining)
    used.extend(round_trick_cards)

    extra = set(used) - expected
    if extra:
        raise AssertionError(f"Cards are incompatible with config: {sorted(extra)}")
    missing = [card_id_ for card_id_ in sorted(expected) if card_id_ not in used]
    round_discarded.extend(missing)

    round_ = TichuRound(
        seed=0,
        dealer=dealer,
        hands=round_hands,
        deck_remaining=round_deck_remaining,
        phase=phase,
        current_player=current_player,
        config=config,
        captured=round_captured,
        discarded=round_discarded,
        pending_exchanges={} if pending_exchanges is None else dict(pending_exchanges),
        tichu_calls={} if tichu_calls is None else dict(tichu_calls),
        has_played_card=list(has_played_card or [False] * 4),
        wish_rank=wish_rank,
        top_combination=top_combination,
        trick_cards=round_trick_cards,
        last_play_player=last_play_player,
        consecutive_passes=consecutive_passes,
        finish_order=list(finish_order),
        bomb_queue=list(bomb_queue),
        bomb_resume_player=bomb_resume_player,
        bomb_close_behavior=bomb_close_behavior,
        pending_dragon_winner=pending_dragon_winner,
        finalize_after_dragon=finalize_after_dragon,
        round_scores=round_scores,
    )
    round_.assert_valid()
    return round_
