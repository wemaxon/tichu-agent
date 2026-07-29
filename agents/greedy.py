from __future__ import annotations

from collections.abc import Sequence

from tichu.actions import (
    Action,
    BombAction,
    DeclareSmallTichu,
    DragonGiftAction,
    ExchangeAction,
    GrandTichuDecision,
    PassAction,
    PlayAction,
)
from tichu.cards import CARD_BY_ID, Special
from tichu.combinations import Combination, CombinationKind
from tichu.observations import RoundObservation


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


class GreedyAgent:
    """Deterministic baseline that plays the cheapest useful legal action."""

    def choose_action(
        self,
        observation: RoundObservation,
        legal_actions: Sequence[Action],
    ) -> Action:
        actions = list(legal_actions)
        if not actions:
            raise RuntimeError("The simulator produced no legal actions")

        phase = observation["phase"]
        if phase == "grand_tichu":
            return _first(actions, GrandTichuDecision(False))
        if phase == "exchange":
            return _choose_exchange(observation, actions)
        if phase == "dragon_gift":
            return _choose_dragon_gift(observation, actions)
        if phase == "bomb_window":
            return _choose_bomb_window(observation, actions)
        if phase == "play":
            return _choose_play(observation, actions)
        return actions[0]


def _choose_exchange(observation: RoundObservation, actions: Sequence[Action]) -> Action:
    player = observation["current_player"]
    if player is None:
        raise RuntimeError("No action is available")

    hand = tuple(card["card_id"] for card in observation["hand"])
    recipients = tuple(other for other in range(4) if other != player)
    selected = tuple(sorted(hand, key=_card_value)[:3])
    preferred = ExchangeAction(tuple(sorted(zip(recipients, selected), key=lambda item: item[0])))
    if preferred in actions:
        return preferred

    exchanges = [action for action in actions if isinstance(action, ExchangeAction)]
    if not exchanges:
        return actions[0]
    return min(exchanges, key=_exchange_key)


def _choose_dragon_gift(observation: RoundObservation, actions: Sequence[Action]) -> Action:
    gifts = [action for action in actions if isinstance(action, DragonGiftAction)]
    if not gifts:
        return actions[0]
    hand_sizes = observation["hand_sizes"]
    return max(gifts, key=lambda action: (hand_sizes[action.recipient], action.recipient))


def _choose_bomb_window(observation: RoundObservation, actions: Sequence[Action]) -> Action:
    bombs = [
        action
        for action in actions
        if isinstance(action, BombAction) and action.combination is not None
    ]
    decline = _find_decline_bomb(actions)
    hand_size = len(observation["hand"])
    finishing_bombs = [
        action
        for action in bombs
        if action.combination is not None and action.combination.length == hand_size
    ]
    if finishing_bombs:
        return min(finishing_bombs, key=lambda action: _combination_cost(action.combination))
    if decline is not None:
        return decline
    if bombs:
        return min(bombs, key=lambda action: _combination_cost(action.combination))
    return actions[0]


def _choose_play(observation: RoundObservation, actions: Sequence[Action]) -> Action:
    plays = [action for action in actions if isinstance(action, PlayAction)]
    non_bomb_plays = [action for action in plays if not action.combination.is_bomb]
    pass_action = _find_pass(actions)
    leading = observation["top_combination"] is None

    if non_bomb_plays:
        return min(
            non_bomb_plays,
            key=lambda action: _lead_play_key(action) if leading else _follow_play_key(action),
        )

    if pass_action is not None:
        return pass_action
    if plays:
        return min(plays, key=lambda action: _combination_cost(action.combination))

    declaration = _find_small_tichu(actions)
    if declaration is not None:
        return declaration
    return actions[0]


def _lead_play_key(action: PlayAction) -> tuple[float, float, int, float, tuple[int, ...], int]:
    combo = action.combination
    return (
        -combo.length,
        _combination_cost(combo),
        _KIND_ORDER[combo.kind],
        combo.primary_rank,
        combo.card_ids,
        0 if action.wish_rank is None else action.wish_rank,
    )


def _follow_play_key(action: PlayAction) -> tuple[float, int, int, float, tuple[int, ...], int]:
    combo = action.combination
    return (
        _combination_cost(combo),
        combo.length,
        _KIND_ORDER[combo.kind],
        combo.primary_rank,
        combo.card_ids,
        0 if action.wish_rank is None else action.wish_rank,
    )


def _exchange_key(action: ExchangeAction) -> tuple[float, tuple[int, ...]]:
    card_ids = tuple(card_id for _, card_id in action.transfers)
    return sum(_card_value(card_id) for card_id in card_ids), card_ids


def _combination_cost(combo: Combination | None) -> float:
    if combo is None:
        return 0.0
    return sum(_card_value(card_id) for card_id in combo.card_ids)


def _card_value(card_id: int) -> float:
    card = CARD_BY_ID[card_id]
    if card.is_normal:
        return float(card.normal_rank) + card.points / 100.0
    if card.special is Special.MAHJONG:
        return 1.0
    if card.special is Special.DOG:
        return 1.25
    if card.special is Special.PHOENIX:
        return 14.5
    if card.special is Special.DRAGON:
        return 15.25
    return 0.0


def _find_pass(actions: Sequence[Action]) -> PassAction | None:
    return next((action for action in actions if isinstance(action, PassAction)), None)


def _find_decline_bomb(actions: Sequence[Action]) -> BombAction | None:
    return next(
        (
            action
            for action in actions
            if isinstance(action, BombAction) and action.combination is None
        ),
        None,
    )


def _find_small_tichu(actions: Sequence[Action]) -> DeclareSmallTichu | None:
    return next((action for action in actions if isinstance(action, DeclareSmallTichu)), None)


def _first(actions: Sequence[Action], preferred: Action) -> Action:
    if preferred in actions:
        return preferred
    return actions[0]
