from __future__ import annotations

from collections import Counter
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


class HeuristicAgent:
    """Rule-based baseline with partner-aware passing and conservative bombs."""

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
            return _choose_grand_tichu(observation, actions)
        if phase == "exchange":
            return _choose_exchange(observation, actions)
        if phase == "dragon_gift":
            return _choose_dragon_gift(observation, actions)
        if phase == "bomb_window":
            return _choose_bomb_window(observation, actions)
        if phase == "play":
            return _choose_play(observation, actions)
        return actions[0]


def _choose_grand_tichu(observation: RoundObservation, actions: Sequence[Action]) -> Action:
    preferred = GrandTichuDecision(_hand_strength(_hand_ids(observation)) >= 34.0)
    if preferred in actions:
        return preferred
    return actions[0]


def _choose_exchange(observation: RoundObservation, actions: Sequence[Action]) -> Action:
    player = observation["current_player"]
    if player is None:
        raise RuntimeError("No action is available")

    hand = _hand_ids(observation)
    recipients = tuple(other for other in range(4) if other != player)
    partner = _partner_of(player)
    selected = set(sorted(hand, key=_exchange_opponent_value)[:3])
    transfers: dict[int, int] = {}

    if partner in recipients:
        partner_card = max(selected, key=_exchange_partner_value)
        transfers[partner] = partner_card
        selected.remove(partner_card)

    for recipient in recipients:
        if recipient == partner:
            continue
        card_id = min(selected, key=_exchange_opponent_value)
        transfers[recipient] = card_id
        selected.remove(card_id)

    preferred = ExchangeAction(tuple(sorted(transfers.items())))
    if preferred in actions:
        return preferred

    exchanges = [action for action in actions if isinstance(action, ExchangeAction)]
    if not exchanges:
        return actions[0]
    return min(exchanges, key=lambda action: _exchange_action_key(action, partner))


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
    if not bombs:
        return decline if decline is not None else actions[0]

    current_player = observation["current_player"]
    top_player = _top_player(observation)
    hand_size = len(observation["hand"])
    finishing = [
        action
        for action in bombs
        if action.combination is not None and action.combination.length == hand_size
    ]
    if finishing:
        return min(finishing, key=lambda action: _combination_cost(action.combination))

    if current_player is not None and top_player is not None and _same_team(current_player, top_player):
        return decline if decline is not None else actions[0]

    if _should_bomb_observed_top(observation, top_player):
        return min(bombs, key=lambda action: _bomb_key(action.combination))

    return decline if decline is not None else min(bombs, key=lambda action: _bomb_key(action.combination))


def _choose_play(observation: RoundObservation, actions: Sequence[Action]) -> Action:
    declaration = _find_small_tichu(actions)
    if declaration is not None and _hand_strength(_hand_ids(observation)) >= 60.0:
        return declaration

    plays = [action for action in actions if isinstance(action, PlayAction)]
    pass_action = _find_pass(actions)
    if not plays:
        return pass_action if pass_action is not None else actions[0]

    leading = observation["top_combination"] is None
    if leading:
        return _choose_lead(observation, plays)

    current_player = observation["current_player"]
    top_player = _top_player(observation)
    if (
        pass_action is not None
        and current_player is not None
        and top_player is not None
        and _same_team(current_player, top_player)
    ):
        return pass_action

    hand_size = len(observation["hand"])
    finishing = [action for action in plays if action.combination.length == hand_size]
    if finishing:
        return min(finishing, key=lambda action: _follow_key(action))

    non_bombs = [action for action in plays if not action.combination.is_bomb]
    if non_bombs:
        return min(non_bombs, key=_follow_key)

    if pass_action is not None and not _should_bomb_observed_top(observation, top_player):
        return pass_action
    return min(plays, key=lambda action: _bomb_key(action.combination))


def _choose_lead(observation: RoundObservation, plays: Sequence[PlayAction]) -> PlayAction:
    hand_size = len(observation["hand"])
    finishing = [action for action in plays if action.combination.length == hand_size]
    if finishing:
        return min(finishing, key=_lead_key)

    non_bombs = [action for action in plays if not action.combination.is_bomb]
    if non_bombs:
        return min(non_bombs, key=_lead_key)
    return min(plays, key=lambda action: _bomb_key(action.combination))


def _lead_key(action: PlayAction) -> tuple[float, float, int, float, tuple[int, ...], int]:
    combo = action.combination
    return (
        -combo.length,
        _combination_cost(combo) + _special_use_penalty(combo),
        _KIND_ORDER[combo.kind],
        combo.primary_rank,
        combo.card_ids,
        _wish_key(action),
    )


def _follow_key(action: PlayAction) -> tuple[float, int, int, float, tuple[int, ...], int]:
    combo = action.combination
    return (
        _combination_cost(combo) + _special_use_penalty(combo),
        combo.length,
        _KIND_ORDER[combo.kind],
        combo.primary_rank,
        combo.card_ids,
        _wish_key(action),
    )


def _bomb_key(combo: Combination | None) -> tuple[float, int, float, tuple[int, ...]]:
    if combo is None:
        return 0.0, 0, 0.0, ()
    return _combination_cost(combo), combo.length, combo.primary_rank, combo.card_ids


def _exchange_action_key(action: ExchangeAction, partner: int) -> tuple[float, tuple[int, ...]]:
    score = 0.0
    card_ids: list[int] = []
    for recipient, card_id in action.transfers:
        card_ids.append(card_id)
        if recipient == partner:
            score -= _exchange_partner_value(card_id)
        else:
            score += _exchange_opponent_value(card_id)
    return score, tuple(card_ids)


def _should_bomb_observed_top(observation: RoundObservation, top_player: int | None) -> bool:
    current_player = observation["current_player"]
    if current_player is None or top_player is None or _same_team(current_player, top_player):
        return False

    hand_sizes = observation["hand_sizes"]
    if hand_sizes[top_player] <= 2:
        return True

    top = observation["top_combination"]
    if top is None:
        return False
    return top["kind"] == CombinationKind.SINGLE.value and top["primary_rank"] >= 15.0


def _top_player(observation: RoundObservation) -> int | None:
    if observation["top_combination"] is None:
        return None
    for event in reversed(observation["public_history"]):
        event_type = event.get("type")
        if event_type in {"play", "bomb"}:
            player = event.get("player")
            return player if isinstance(player, int) else None
        if event_type in {"trick_won", "final_trick_won", "dragon_gift", "round_over"}:
            return None
    return None


def _hand_strength(card_ids: Sequence[int]) -> float:
    ranks = [
        CARD_BY_ID[card_id].normal_rank
        for card_id in card_ids
        if CARD_BY_ID[card_id].is_normal
    ]
    counts = Counter(ranks)
    score = 0.0
    for card_id in card_ids:
        card = CARD_BY_ID[card_id]
        if card.is_normal:
            score += max(0.0, (card.normal_rank - 10) * 1.4)
        elif card.special is Special.DRAGON:
            score += 6.5
        elif card.special is Special.PHOENIX:
            score += 5.5
        elif card.special is Special.MAHJONG:
            score += 1.5
        elif card.special is Special.DOG:
            score -= 0.5

    for rank, count in counts.items():
        if count >= 2 and rank >= 10:
            score += (count - 1) * (rank - 8)
        if count == 4:
            score += 4.0
    return score


def _combination_cost(combo: Combination | None) -> float:
    if combo is None:
        return 0.0
    return sum(_card_value(card_id) for card_id in combo.card_ids)


def _special_use_penalty(combo: Combination) -> float:
    penalty = 0.0
    if combo.contains_special(Special.PHOENIX):
        penalty += 3.0
    if combo.contains_special(Special.DRAGON):
        penalty += 2.0
    if combo.is_bomb:
        penalty += 20.0
    return penalty


def _card_value(card_id: int) -> float:
    card = CARD_BY_ID[card_id]
    if card.is_normal:
        return float(card.normal_rank) + card.points / 50.0
    if card.special is Special.MAHJONG:
        return 1.0
    if card.special is Special.DOG:
        return 1.25
    if card.special is Special.PHOENIX:
        return 14.5
    if card.special is Special.DRAGON:
        return 15.5
    return 0.0


def _exchange_opponent_value(card_id: int) -> float:
    return _card_value(card_id) + _special_exchange_penalty(card_id)


def _exchange_partner_value(card_id: int) -> float:
    return _card_value(card_id) + _special_partner_bonus(card_id)


def _special_exchange_penalty(card_id: int) -> float:
    card = CARD_BY_ID[card_id]
    if card.special is Special.DOG:
        return 2.0
    if card.special in {Special.PHOENIX, Special.DRAGON, Special.MAHJONG}:
        return 20.0
    return 0.0


def _special_partner_bonus(card_id: int) -> float:
    card = CARD_BY_ID[card_id]
    if card.special is Special.DOG:
        return 4.0
    if card.special is Special.PHOENIX:
        return 2.0
    if card.special is Special.DRAGON:
        return 1.0
    return 0.0


def _hand_ids(observation: RoundObservation) -> tuple[int, ...]:
    return tuple(card["card_id"] for card in observation["hand"])


def _partner_of(player: int) -> int:
    return (player + 2) % 4


def _same_team(left: int, right: int) -> bool:
    return left % 2 == right % 2


def _wish_key(action: PlayAction) -> int:
    if not action.combination.contains_special(Special.MAHJONG):
        return 0
    return -1 if action.wish_rank is None else -action.wish_rank


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
