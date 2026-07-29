from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
from tichu.observations import CombinationObservation, PhaseName, RoundObservation

FeatureVector: TypeAlias = tuple[float, ...]
T = TypeVar("T")

CARD_COUNT = len(CARD_CLASS_NAMES)
PLAYER_COUNT = 4
TEAM_COUNT = 2
MAX_HAND_SIZE = 14.0
MAX_PLAY_RANK = 15.0
MAX_NORMAL_RANK = 14.0
SCORE_SCALE = 1000.0

PHASES: tuple[PhaseName, ...] = (
    "grand_tichu",
    "exchange",
    "play",
    "bomb_window",
    "dragon_gift",
    "round_over",
)
CONFIG_KEYS: tuple[str, ...] = (
    "grand_tichu",
    "small_tichu",
    "card_exchange",
    "bombs",
    "wishes",
    "special_cards",
)
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
OBS_PUBLIC_CARD_MASK_OFFSET = OBS_HAND_MASK_OFFSET + CARD_COUNT
OBS_PHASE_OFFSET = OBS_PUBLIC_CARD_MASK_OFFSET + CARD_COUNT
OBS_CURRENT_PLAYER_OFFSET = OBS_PHASE_OFFSET + len(PHASES)
OBS_PLAYER_OFFSET = OBS_CURRENT_PLAYER_OFFSET + PLAYER_COUNT + 1
OBS_TEAM_OFFSET = OBS_PLAYER_OFFSET + PLAYER_COUNT
OBS_HAND_SIZES_OFFSET = OBS_TEAM_OFFSET + TEAM_COUNT
OBS_CONFIG_OFFSET = OBS_HAND_SIZES_OFFSET + PLAYER_COUNT
OBS_TOP_COMBINATION_OFFSET = OBS_CONFIG_OFFSET + len(CONFIG_KEYS)
OBS_WISH_RANK_OFFSET = OBS_TOP_COMBINATION_OFFSET + COMBINATION_FEATURE_SIZE
OBS_FINISH_ORDER_OFFSET = OBS_WISH_RANK_OFFSET + len(WISH_RANKS)
OBS_TICHU_CALLS_OFFSET = OBS_FINISH_ORDER_OFFSET + PLAYER_COUNT
OBS_ROUND_SCORES_OFFSET = OBS_TICHU_CALLS_OFFSET + PLAYER_COUNT
OBSERVATION_FEATURE_SIZE = OBS_ROUND_SCORES_OFFSET + TEAM_COUNT

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
    hand_ids = (card["card_id"] for card in observation["hand"])
    return (
        encode_card_mask(hand_ids)
        + _public_card_mask(observation["public_history"])
        + _one_hot(observation["phase"], PHASES)
        + _optional_player_features(observation["current_player"])
        + _player_features(observation["player"])
        + _one_hot(observation["team"], range(TEAM_COUNT))
        + _scaled_values(observation["hand_sizes"], MAX_HAND_SIZE)
        + _config_features(observation["config"])
        + _encode_combination(observation["top_combination"])
        + _optional_rank_features(observation["wish_rank"])
        + _finish_order_features(observation["finish_order"])
        + _tichu_call_features(observation["tichu_calls"])
        + _round_score_features(observation["round_scores"])
    )


def encode_action(action: Action) -> FeatureVector:
    action_type = _action_type(action)
    combination = _action_combination(action)
    card_ids = _action_card_ids(action)
    return (
        _one_hot(action_type, ACTION_TYPES)
        + encode_card_mask(card_ids)
        + _encode_combination(combination)
        + _optional_rank_features(_action_wish_rank(action))
        + _player_features(_action_target_player(action))
        + (_action_tichu_value(action),)
        + _exchange_transfer_features(action)
    )


def encode_legal_actions(actions: Sequence[Action]) -> tuple[EncodedAction, ...]:
    return tuple(EncodedAction(action, encode_action(action)) for action in actions)


def encode_card_mask(card_ids: Iterable[int]) -> FeatureVector:
    values = [0.0] * CARD_COUNT
    for raw_card_id in card_ids:
        card_id = int(raw_card_id)
        if card_id < 0 or card_id >= CARD_COUNT:
            raise ValueError(f"card_id must be between 0 and {CARD_COUNT - 1}: {card_id}")
        values[card_id] = 1.0
    return tuple(values)


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


def _optional_rank_features(rank: int | None) -> FeatureVector:
    if rank is not None and rank not in range(2, 15):
        raise ValueError(f"rank must be between 2 and 14 or None: {rank}")
    return _one_hot(rank, WISH_RANKS)


def _scaled_values(values: Iterable[int | float], scale: float) -> FeatureVector:
    return tuple(float(value) / scale for value in values)


def _config_features(config: Mapping[str, bool]) -> FeatureVector:
    return tuple(1.0 if bool(config.get(key, False)) else 0.0 for key in CONFIG_KEYS)


def _round_score_features(scores: tuple[int, int] | None) -> FeatureVector:
    if scores is None:
        return (0.0, 0.0)
    if len(scores) != TEAM_COUNT:
        raise ValueError("round_scores must contain exactly two values")
    return _scaled_values(scores, SCORE_SCALE)


def _finish_order_features(finish_order: Sequence[int]) -> FeatureVector:
    values = [0.0] * PLAYER_COUNT
    for position, player in enumerate(finish_order, start=1):
        if player not in range(PLAYER_COUNT):
            raise ValueError(f"finish_order contains invalid player: {player}")
        values[player] = float(position) / PLAYER_COUNT
    return tuple(values)


def _tichu_call_features(calls: Mapping[int, int]) -> FeatureVector:
    values: list[float] = []
    for player in range(PLAYER_COUNT):
        raw = calls.get(player)
        values.append(0.0 if raw is None else float(raw) / 200.0)
    return tuple(values)


def _public_card_mask(public_history: Sequence[Mapping[str, object]]) -> FeatureVector:
    card_ids: list[int] = []
    for event in public_history:
        combination = event.get("combination")
        if isinstance(combination, Mapping):
            raw_card_ids = combination.get("card_ids")
            if raw_card_ids is not None:
                card_ids.extend(_card_ids_from_object(raw_card_ids, "public combination card_ids"))
    return encode_card_mask(card_ids)


def _encode_combination(
    combination: Combination | CombinationObservation | None,
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
    combination: Combination | CombinationObservation,
) -> tuple[int, ...]:
    if isinstance(combination, Combination):
        return combination.card_ids
    return tuple(combination["card_ids"])


def _combination_kind(
    combination: Combination | CombinationObservation,
) -> str:
    if isinstance(combination, Combination):
        return combination.kind.value
    kind = combination["kind"]
    if kind not in COMBINATION_KIND_INDEX:
        raise ValueError(f"Unsupported combination kind: {kind!r}")
    return kind


def _combination_primary_rank(
    combination: Combination | CombinationObservation,
) -> float:
    if isinstance(combination, Combination):
        return float(combination.primary_rank)
    return float(combination["primary_rank"])


def _combination_length(
    combination: Combination | CombinationObservation,
) -> int:
    if isinstance(combination, Combination):
        return combination.length
    return int(combination["length"])


def _combination_phoenix_as(
    combination: Combination | CombinationObservation,
) -> int | None:
    if isinstance(combination, Combination):
        return combination.phoenix_as
    return combination["phoenix_as"]


def _card_ids_from_object(raw: object, context: str) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{context} must be a sequence")

    card_ids: list[int] = []
    for value in raw:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{context} must contain only integer card IDs")
        card_ids.append(value)
    return tuple(card_ids)


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


def _action_target_player(action: Action) -> int | None:
    if isinstance(action, DragonGiftAction):
        return action.recipient
    return None


def _action_tichu_value(action: Action) -> float:
    if isinstance(action, GrandTichuDecision):
        return 1.0 if action.call else 0.0
    if isinstance(action, DeclareSmallTichu):
        return 0.5
    return 0.0


def _exchange_transfer_features(action: Action) -> FeatureVector:
    values = [0.0] * (PLAYER_COUNT * CARD_COUNT)
    if not isinstance(action, ExchangeAction):
        return tuple(values)

    for recipient, card_id in action.transfers:
        if recipient not in range(PLAYER_COUNT):
            raise ValueError(f"recipient must be between 0 and {PLAYER_COUNT - 1}: {recipient}")
        if card_id < 0 or card_id >= CARD_COUNT:
            raise ValueError(f"card_id must be between 0 and {CARD_COUNT - 1}: {card_id}")
        values[recipient * CARD_COUNT + card_id] = 1.0
    return tuple(values)
