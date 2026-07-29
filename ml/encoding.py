from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

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
from tichu.cards import CARD_CLASS_NAMES
from tichu.combinations import Combination, CombinationKind
from tichu.observations import BombCloseBehaviorName, PhaseName, RoundObservation

from .policy_view import (
    PLAYER_COUNT,
    POLICY_CONFIG_KEYS,
    PolicyCombination,
    PolicyExchangeCard,
    PolicyObservation,
    policy_view,
    relative_player,
)

FeatureVector: TypeAlias = tuple[float, ...]
T = TypeVar("T")

CARD_COUNT = len(CARD_CLASS_NAMES)
MAX_HAND_SIZE = 14.0
MAX_PLAY_RANK = 15.0
MAX_NORMAL_RANK = 14.0
MAX_CONSECUTIVE_PASSES = 3.0
CAPTURED_POINTS_SCALE = 100.0

PHASES: tuple[PhaseName, ...] = (
    "grand_tichu",
    "exchange",
    "play",
    "bomb_window",
    "dragon_gift",
    "round_over",
)
CONFIG_KEYS = POLICY_CONFIG_KEYS
COMBINATION_KINDS: tuple[str, ...] = tuple(kind.value for kind in CombinationKind)
ACTION_TYPES: tuple[str, ...] = (
    "grand_tichu",
    "exchange",
    "small_tichu",
    "play",
    "pass",
    "bomb",
    "dragon_gift",
)
WISH_RANKS: tuple[int | None, ...] = (None,) + tuple(range(2, 15))
BOMB_CLOSE_BEHAVIORS: tuple[BombCloseBehaviorName | None, ...] = (
    None,
    "continue",
    "resolve_trick",
)

PHASE_INDEX = {phase: index for index, phase in enumerate(PHASES)}
COMBINATION_KIND_INDEX = {
    kind: index
    for index, kind in enumerate(COMBINATION_KINDS)
}
ACTION_TYPE_INDEX = {
    action_type: index
    for index, action_type in enumerate(ACTION_TYPES)
}

COMBINATION_CARD_MASK_OFFSET = 0
COMBINATION_KIND_OFFSET = COMBINATION_CARD_MASK_OFFSET + CARD_COUNT
COMBINATION_PRESENT_OFFSET = COMBINATION_KIND_OFFSET + len(COMBINATION_KINDS)
COMBINATION_PRIMARY_RANK_OFFSET = COMBINATION_PRESENT_OFFSET + 1
COMBINATION_LENGTH_OFFSET = COMBINATION_PRIMARY_RANK_OFFSET + 1
COMBINATION_PHOENIX_PRESENT_OFFSET = COMBINATION_LENGTH_OFFSET + 1
COMBINATION_PHOENIX_RANK_OFFSET = COMBINATION_PHOENIX_PRESENT_OFFSET + 1
COMBINATION_FEATURE_SIZE = COMBINATION_PHOENIX_RANK_OFFSET + 1

OBS_HAND_MASK_OFFSET = 0
OBS_PLAYED_CARD_MASKS_OFFSET = OBS_HAND_MASK_OFFSET + CARD_COUNT
OBS_CAPTURED_CARD_MASKS_OFFSET = OBS_PLAYED_CARD_MASKS_OFFSET + PLAYER_COUNT * CARD_COUNT
OBS_CURRENT_TRICK_MASK_OFFSET = OBS_CAPTURED_CARD_MASKS_OFFSET + PLAYER_COUNT * CARD_COUNT
OBS_PHASE_OFFSET = OBS_CURRENT_TRICK_MASK_OFFSET + CARD_COUNT
OBS_CURRENT_PLAYER_OFFSET = OBS_PHASE_OFFSET + len(PHASES)
OBS_DEALER_OFFSET = OBS_CURRENT_PLAYER_OFFSET + PLAYER_COUNT + 1
OBS_HAND_SIZES_OFFSET = OBS_DEALER_OFFSET + PLAYER_COUNT
OBS_CONFIG_OFFSET = OBS_HAND_SIZES_OFFSET + PLAYER_COUNT
OBS_TOP_COMBINATION_OFFSET = OBS_CONFIG_OFFSET + len(CONFIG_KEYS)
OBS_TOP_PLAYER_OFFSET = OBS_TOP_COMBINATION_OFFSET + COMBINATION_FEATURE_SIZE
OBS_CONSECUTIVE_PASSES_OFFSET = OBS_TOP_PLAYER_OFFSET + PLAYER_COUNT + 1
OBS_BOMB_QUEUE_OFFSET = OBS_CONSECUTIVE_PASSES_OFFSET + 1
OBS_BOMB_RESUME_PLAYER_OFFSET = OBS_BOMB_QUEUE_OFFSET + PLAYER_COUNT * (PLAYER_COUNT + 1)
OBS_BOMB_CLOSE_BEHAVIOR_OFFSET = OBS_BOMB_RESUME_PLAYER_OFFSET + PLAYER_COUNT + 1
OBS_WISH_RANK_OFFSET = OBS_BOMB_CLOSE_BEHAVIOR_OFFSET + len(BOMB_CLOSE_BEHAVIORS)
OBS_FINISH_ORDER_OFFSET = OBS_WISH_RANK_OFFSET + len(WISH_RANKS)
OBS_TICHU_CALLS_OFFSET = OBS_FINISH_ORDER_OFFSET + PLAYER_COUNT
OBS_HAS_PLAYED_CARD_OFFSET = OBS_TICHU_CALLS_OFFSET + PLAYER_COUNT
OBS_CAPTURED_POINTS_OFFSET = OBS_HAS_PLAYED_CARD_OFFSET + PLAYER_COUNT
OBS_PRIVATE_EXCHANGE_PRESENT_OFFSET = OBS_CAPTURED_POINTS_OFFSET + PLAYER_COUNT
OBS_EXCHANGE_SENT_OFFSET = OBS_PRIVATE_EXCHANGE_PRESENT_OFFSET + 1
OBS_EXCHANGE_RECEIVED_OFFSET = OBS_EXCHANGE_SENT_OFFSET + PLAYER_COUNT * CARD_COUNT
OBSERVATION_FEATURE_SIZE = OBS_EXCHANGE_RECEIVED_OFFSET + PLAYER_COUNT * CARD_COUNT

ACTION_TYPE_OFFSET = 0
ACTION_CARD_MASK_OFFSET = ACTION_TYPE_OFFSET + len(ACTION_TYPES)
ACTION_COMBINATION_OFFSET = ACTION_CARD_MASK_OFFSET + CARD_COUNT
ACTION_WISH_RANK_OFFSET = ACTION_COMBINATION_OFFSET + COMBINATION_FEATURE_SIZE
ACTION_TARGET_PLAYER_OFFSET = ACTION_WISH_RANK_OFFSET + len(WISH_RANKS)
ACTION_TICHU_VALUE_OFFSET = ACTION_TARGET_PLAYER_OFFSET + PLAYER_COUNT
ACTION_TRANSFER_MASK_OFFSET = ACTION_TICHU_VALUE_OFFSET + 1
ACTION_FEATURE_SIZE = ACTION_TRANSFER_MASK_OFFSET + PLAYER_COUNT * CARD_COUNT


@dataclass(frozen=True, slots=True)
class EncodedAction:
    action: Action
    features: FeatureVector


def encode_observation(observation: RoundObservation) -> FeatureVector:
    return encode_policy_observation(policy_view(observation))


def encode_policy_observation(observation: PolicyObservation) -> FeatureVector:
    exchange = observation.private_exchange
    return (
        encode_card_mask(observation.hand_card_ids)
        + _player_card_masks(observation.played_card_ids)
        + _player_card_masks(observation.captured_card_ids)
        + encode_card_mask(observation.current_trick_card_ids)
        + _one_hot(observation.phase, PHASES)
        + _optional_player_features(observation.current_player)
        + _player_features(observation.dealer)
        + _scaled_values(observation.hand_sizes, MAX_HAND_SIZE)
        + tuple(1.0 if enabled else 0.0 for enabled in observation.config)
        + _encode_combination(observation.top_combination)
        + _optional_player_features(observation.top_player)
        + (float(observation.consecutive_passes) / MAX_CONSECUTIVE_PASSES,)
        + _player_sequence_features(observation.bomb_queue)
        + _optional_player_features(observation.bomb_resume_player)
        + _one_hot(observation.bomb_close_behavior, BOMB_CLOSE_BEHAVIORS)
        + _optional_rank_features(observation.wish_rank)
        + _finish_order_features(observation.finish_order)
        + _tichu_call_features(observation.tichu_calls)
        + tuple(1.0 if played else 0.0 for played in observation.has_played_card)
        + _scaled_values(observation.captured_points, CAPTURED_POINTS_SCALE)
        + (1.0 if exchange is not None else 0.0,)
        + _exchange_card_features(None if exchange is None else exchange.sent)
        + _exchange_card_features(None if exchange is None else exchange.received)
    )


def encode_action(action: Action, *, observer: int) -> FeatureVector:
    if observer not in range(PLAYER_COUNT):
        raise ValueError(f"observer must be between 0 and {PLAYER_COUNT - 1}: {observer}")
    action_type = _action_type(action)
    combination = _action_combination(action)
    card_ids = _action_card_ids(action)
    return (
        _one_hot(action_type, ACTION_TYPES)
        + encode_card_mask(card_ids)
        + _encode_combination(combination)
        + _optional_rank_features(_action_wish_rank(action))
        + _player_features(_action_target_player(action, observer))
        + (_action_tichu_value(action),)
        + _exchange_transfer_features(action, observer)
    )


def encode_legal_actions(
    actions: Sequence[Action],
    *,
    observer: int,
) -> tuple[EncodedAction, ...]:
    return tuple(
        EncodedAction(action, encode_action(action, observer=observer))
        for action in actions
    )


def encode_card_mask(card_ids: Iterable[int]) -> FeatureVector:
    values = [0.0] * CARD_COUNT
    for raw_card_id in card_ids:
        card_id = int(raw_card_id)
        if card_id < 0 or card_id >= CARD_COUNT:
            raise ValueError(f"card_id must be between 0 and {CARD_COUNT - 1}: {card_id}")
        values[card_id] = 1.0
    return tuple(values)


def _player_card_masks(card_ids: Sequence[Iterable[int]]) -> FeatureVector:
    if len(card_ids) != PLAYER_COUNT:
        raise ValueError("Player card groups must contain exactly four entries")
    return tuple(
        value
        for player_cards in card_ids
        for value in encode_card_mask(player_cards)
    )


def _one_hot(value: T, choices: Iterable[T]) -> FeatureVector:
    return tuple(1.0 if value == choice else 0.0 for choice in choices)


def _player_features(player: int | None) -> FeatureVector:
    if player is None:
        return (0.0,) * PLAYER_COUNT
    if player not in range(PLAYER_COUNT):
        raise ValueError(f"player must be between 0 and {PLAYER_COUNT - 1}: {player}")
    return _one_hot(player, range(PLAYER_COUNT))


def _optional_player_features(player: int | None) -> FeatureVector:
    if player is None:
        return (0.0,) * PLAYER_COUNT + (1.0,)
    return _player_features(player) + (0.0,)


def _player_sequence_features(players: Sequence[int]) -> FeatureVector:
    if len(players) > PLAYER_COUNT:
        raise ValueError("A player sequence cannot contain more than four entries")
    padded: tuple[int | None, ...] = tuple(players) + (None,) * (
        PLAYER_COUNT - len(players)
    )
    return tuple(
        feature
        for player in padded
        for feature in _optional_player_features(player)
    )


def _optional_rank_features(rank: int | None) -> FeatureVector:
    if rank is not None and rank not in range(2, 15):
        raise ValueError(f"rank must be between 2 and 14 or None: {rank}")
    return _one_hot(rank, WISH_RANKS)


def _scaled_values(values: Iterable[int | float], scale: float) -> FeatureVector:
    return tuple(float(value) / scale for value in values)


def _finish_order_features(finish_order: Sequence[int]) -> FeatureVector:
    values = [0.0] * PLAYER_COUNT
    for position, player in enumerate(finish_order, start=1):
        if player not in range(PLAYER_COUNT):
            raise ValueError(f"finish_order contains invalid player: {player}")
        values[player] = float(position) / PLAYER_COUNT
    return tuple(values)


def _tichu_call_features(calls: Sequence[int | None]) -> FeatureVector:
    if len(calls) != PLAYER_COUNT:
        raise ValueError("Tichu calls must contain exactly four entries")
    return tuple(0.0 if value is None else float(value) / 200.0 for value in calls)


def _exchange_card_features(
    transfers: Sequence[PolicyExchangeCard] | None,
) -> FeatureVector:
    values = [0.0] * (PLAYER_COUNT * CARD_COUNT)
    if transfers is None:
        return tuple(values)
    for transfer in transfers:
        if transfer.player not in range(PLAYER_COUNT):
            raise ValueError(f"Invalid relative exchange player: {transfer.player}")
        if transfer.card_id not in range(CARD_COUNT):
            raise ValueError(f"Invalid exchange card ID: {transfer.card_id}")
        values[transfer.player * CARD_COUNT + transfer.card_id] = 1.0
    return tuple(values)


def _encode_combination(
    combination: Combination | PolicyCombination | None,
) -> FeatureVector:
    if combination is None:
        return (0.0,) * COMBINATION_FEATURE_SIZE

    card_ids = _combination_card_ids(combination)
    kind = _combination_kind(combination)
    primary_rank = _combination_primary_rank(combination)
    length = _combination_length(combination)
    phoenix_as = _combination_phoenix_as(combination)

    return (
        encode_card_mask(card_ids)
        + _one_hot(kind, COMBINATION_KINDS)
        + (1.0,)
        + (primary_rank / MAX_PLAY_RANK,)
        + (float(length) / MAX_HAND_SIZE,)
        + (1.0 if phoenix_as is not None else 0.0,)
        + (0.0 if phoenix_as is None else float(phoenix_as) / MAX_NORMAL_RANK,)
    )


def _combination_card_ids(
    combination: Combination | PolicyCombination,
) -> tuple[int, ...]:
    return combination.card_ids


def _combination_kind(
    combination: Combination | PolicyCombination,
) -> str:
    if isinstance(combination, Combination):
        return combination.kind.value
    kind = combination.kind
    if kind not in COMBINATION_KIND_INDEX:
        raise ValueError(f"Unsupported combination kind: {kind!r}")
    return kind


def _combination_primary_rank(
    combination: Combination | PolicyCombination,
) -> float:
    return float(combination.primary_rank)


def _combination_length(
    combination: Combination | PolicyCombination,
) -> int:
    return int(combination.length)


def _combination_phoenix_as(
    combination: Combination | PolicyCombination,
) -> int | None:
    return combination.phoenix_as


def _action_type(action: Action) -> str:
    if isinstance(action, GrandTichuDecision):
        return "grand_tichu"
    if isinstance(action, ExchangeAction):
        return "exchange"
    if isinstance(action, DeclareSmallTichu):
        return "small_tichu"
    if isinstance(action, PlayAction):
        return "play"
    if isinstance(action, PassAction):
        return "pass"
    if isinstance(action, BombAction):
        return "bomb"
    if isinstance(action, DragonGiftAction):
        return "dragon_gift"
    raise TypeError(f"Unsupported action: {type(action)!r}")


def _action_card_ids(action: Action) -> tuple[int, ...]:
    if isinstance(action, ExchangeAction):
        return tuple(card_id for _, card_id in action.transfers)
    combination = _action_combination(action)
    return () if combination is None else combination.card_ids


def _action_combination(action: Action) -> Combination | None:
    if isinstance(action, PlayAction):
        return action.combination
    if isinstance(action, BombAction):
        return action.combination
    return None


def _action_wish_rank(action: Action) -> int | None:
    if isinstance(action, PlayAction):
        return action.wish_rank
    return None


def _action_target_player(action: Action, observer: int) -> int | None:
    if isinstance(action, DragonGiftAction):
        return relative_player(action.recipient, observer)
    return None


def _action_tichu_value(action: Action) -> float:
    if isinstance(action, GrandTichuDecision):
        return 1.0 if action.call else 0.0
    if isinstance(action, DeclareSmallTichu):
        return 0.5
    return 0.0


def _exchange_transfer_features(action: Action, observer: int) -> FeatureVector:
    values = [0.0] * (PLAYER_COUNT * CARD_COUNT)
    if not isinstance(action, ExchangeAction):
        return tuple(values)

    for recipient, card_id in action.transfers:
        relative_recipient = relative_player(recipient, observer)
        if card_id < 0 or card_id >= CARD_COUNT:
            raise ValueError(f"card_id must be between 0 and {CARD_COUNT - 1}: {card_id}")
        values[relative_recipient * CARD_COUNT + card_id] = 1.0
    return tuple(values)
