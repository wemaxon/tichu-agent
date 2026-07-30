from __future__ import annotations

import pytest

from ml.encoding import (
    ACTION_CARD_MASK_OFFSET,
    ACTION_COMBINATION_OFFSET,
    ACTION_FEATURE_SIZE,
    ACTION_TARGET_PLAYER_OFFSET,
    ACTION_TICHU_VALUE_OFFSET,
    ACTION_TRANSFER_MASK_OFFSET,
    ACTION_TYPE_OFFSET,
    ACTION_TYPE_INDEX,
    CARD_COUNT,
    COMBINATION_PRESENT_OFFSET,
    OBS_HAND_MASK_OFFSET,
    OBS_PHASE_OFFSET,
    OBSERVATION_FEATURE_SIZE,
    PHASE_INDEX,
    WISH_RANKS,
    ACTION_WISH_RANK_OFFSET,
    encode_action,
    encode_card_mask,
    encode_legal_actions,
    encode_observation,
)
from tichu.actions import (
    BombAction,
    DeclareSmallTichu,
    DragonGiftAction,
    ExchangeAction,
    GrandTichuDecision,
    PlayAction,
)
from tichu.cards import Special
from tichu.combinations import CombinationKind
from tichu.round import TichuRound

from .helpers import card_id, combo_of_kind, normal_ids


def test_card_mask_uses_canonical_card_ids() -> None:
    mask = encode_card_mask((0, 26, 55))

    assert len(mask) == CARD_COUNT
    assert mask[0] == 1.0
    assert mask[26] == 1.0
    assert mask[55] == 1.0
    assert sum(mask) == 3.0


def test_card_mask_rejects_unknown_card_id() -> None:
    with pytest.raises(ValueError, match="card_id"):
        encode_card_mask((56,))


def test_observation_encoder_has_stable_shape_and_hand_mask() -> None:
    round_ = TichuRound.new(seed=0, dealer=0)
    assert round_.current_player is not None
    observation = round_.observation(round_.current_player)

    features = encode_observation(observation)
    hand_mask = features[OBS_HAND_MASK_OFFSET:OBS_HAND_MASK_OFFSET + CARD_COUNT]

    assert len(features) == OBSERVATION_FEATURE_SIZE
    assert sum(hand_mask) == float(len(observation["hand"]))
    for card in observation["hand"]:
        assert hand_mask[card["card_id"]] == 1.0
    assert features[OBS_PHASE_OFFSET + PHASE_INDEX[observation["phase"]]] == 1.0


def test_legal_action_encoder_preserves_actions_and_shape() -> None:
    round_ = TichuRound.new(seed=1, dealer=0)
    assert round_.current_player is not None
    actions = round_.legal_actions()

    encoded = encode_legal_actions(actions, observer=round_.current_player)

    assert tuple(item.action for item in encoded) == actions
    assert all(len(item.features) == ACTION_FEATURE_SIZE for item in encoded)


def test_play_action_encoder_marks_type_cards_combination_and_wish() -> None:
    mahjong = combo_of_kind(
        (card_id(special=Special.MAHJONG),),
        CombinationKind.SINGLE,
    )
    action = PlayAction(mahjong, wish_rank=5)

    features = encode_action(action, observer=0)

    assert len(features) == ACTION_FEATURE_SIZE
    assert features[ACTION_TYPE_OFFSET + ACTION_TYPE_INDEX["play"]] == 1.0
    assert features[ACTION_CARD_MASK_OFFSET + card_id(special=Special.MAHJONG)] == 1.0
    assert features[ACTION_COMBINATION_OFFSET + COMBINATION_PRESENT_OFFSET] == 1.0
    assert features[ACTION_WISH_RANK_OFFSET + WISH_RANKS.index(5)] == 1.0


def test_exchange_action_encoder_marks_recipient_card_matrix() -> None:
    action = ExchangeAction(((1, 0), (2, 26), (3, 55)))

    features = encode_action(action, observer=0)

    assert features[ACTION_TYPE_OFFSET + ACTION_TYPE_INDEX["exchange"]] == 1.0
    assert features[ACTION_CARD_MASK_OFFSET + 0] == 1.0
    assert features[ACTION_CARD_MASK_OFFSET + 26] == 1.0
    assert features[ACTION_CARD_MASK_OFFSET + 55] == 1.0
    assert features[ACTION_TRANSFER_MASK_OFFSET + 1 * CARD_COUNT + 0] == 1.0
    assert features[ACTION_TRANSFER_MASK_OFFSET + 2 * CARD_COUNT + 26] == 1.0
    assert features[ACTION_TRANSFER_MASK_OFFSET + 3 * CARD_COUNT + 55] == 1.0


def test_on_turn_and_out_of_turn_bombs_use_stable_distinct_action_types() -> None:
    bomb = combo_of_kind(normal_ids(8), CombinationKind.FOUR_BOMB)

    on_turn = encode_action(PlayAction(bomb), observer=0)
    out_of_turn = encode_action(BombAction(bomb), observer=0)
    decline = encode_action(BombAction(None), observer=0)

    assert len(on_turn) == ACTION_FEATURE_SIZE
    assert len(out_of_turn) == ACTION_FEATURE_SIZE
    assert len(decline) == ACTION_FEATURE_SIZE
    assert on_turn[ACTION_TYPE_OFFSET + ACTION_TYPE_INDEX["play"]] == 1.0
    assert out_of_turn[ACTION_TYPE_OFFSET + ACTION_TYPE_INDEX["bomb"]] == 1.0
    assert decline[ACTION_TYPE_OFFSET + ACTION_TYPE_INDEX["bomb"]] == 1.0
    assert out_of_turn[ACTION_COMBINATION_OFFSET + COMBINATION_PRESENT_OFFSET] == 1.0
    assert decline[ACTION_COMBINATION_OFFSET + COMBINATION_PRESENT_OFFSET] == 0.0


def test_tichu_and_dragon_action_specific_fields() -> None:
    grand_call = encode_action(GrandTichuDecision(True), observer=0)
    grand_decline = encode_action(GrandTichuDecision(False), observer=0)
    small_tichu = encode_action(DeclareSmallTichu(), observer=0)
    dragon_gift = encode_action(DragonGiftAction(3), observer=0)

    assert grand_call[ACTION_TICHU_VALUE_OFFSET] == 1.0
    assert grand_decline[ACTION_TICHU_VALUE_OFFSET] == 0.0
    assert small_tichu[ACTION_TICHU_VALUE_OFFSET] == 0.5
    assert dragon_gift[ACTION_TARGET_PLAYER_OFFSET + 3] == 1.0
