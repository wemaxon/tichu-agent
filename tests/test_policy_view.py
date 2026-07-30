from __future__ import annotations

from dataclasses import replace

from ml import (
    OBSERVATION_FEATURE_SIZE,
    POLICY_CONFIG_KEYS,
    PolicyExchange,
    PolicyExchangeCard,
    absolute_player,
    encode_action,
    encode_policy_observation,
    policy_view,
    relative_player,
    rotate_player_values,
)
from tichu.actions import (
    DragonGiftAction,
    ExchangeAction,
    GrandTichuDecision,
    PlayAction,
)
from tichu.config import TichuRoundConfig
from tichu.round import Phase, TichuRound


def _play_only_config() -> TichuRoundConfig:
    return TichuRoundConfig(
        grand_tichu=False,
        small_tichu=False,
        card_exchange=False,
        bombs=False,
        wishes=False,
        special_cards=False,
    )


def test_player_coordinates_round_trip_and_rotate() -> None:
    for observer in range(4):
        for player in range(4):
            relative = relative_player(player, observer)
            assert absolute_player(relative, observer) == player

    assert rotate_player_values(("p0", "p1", "p2", "p3"), 2) == (
        "p2",
        "p3",
        "p0",
        "p1",
    )


def test_bomb_timing_modes_share_one_stable_observation_schema() -> None:
    turn_only = TichuRound.new(seed=2, dealer=1)
    interrupt = TichuRound.new(
        seed=2,
        dealer=1,
        config=TichuRoundConfig(out_of_turn_bombs=True),
    )
    assert turn_only.current_player is not None
    assert interrupt.current_player is not None

    turn_only_view = policy_view(turn_only.observation(turn_only.current_player))
    interrupt_view = policy_view(interrupt.observation(interrupt.current_player))
    turn_only_features = encode_policy_observation(turn_only_view)
    interrupt_features = encode_policy_observation(interrupt_view)
    timing_index = POLICY_CONFIG_KEYS.index("out_of_turn_bombs")

    assert not turn_only_view.config[timing_index]
    assert interrupt_view.config[timing_index]
    assert len(turn_only_features) == OBSERVATION_FEATURE_SIZE
    assert len(interrupt_features) == OBSERVATION_FEATURE_SIZE
    assert turn_only_features != interrupt_features


def test_policy_view_canonicalizes_all_player_indexed_fields() -> None:
    round_ = TichuRound.new(seed=3, dealer=1)
    observation = round_.observation(3)
    view = policy_view(observation)

    assert view.current_player == relative_player(round_.current_player, 3)
    assert view.dealer == 2
    assert view.hand_sizes == tuple(
        len(round_.hands[(3 + relative) % 4])
        for relative in range(4)
    )
    assert view.hand_card_ids == round_.hands[3]


def test_strategically_distinct_public_states_have_distinct_encodings() -> None:
    round_ = TichuRound.new(seed=0, dealer=0, config=_play_only_config())
    view = policy_view(round_.observation(round_.current_player))

    top_from_left = replace(view, top_player=1)
    top_from_partner = replace(view, top_player=2)
    assert encode_policy_observation(top_from_left) != encode_policy_observation(
        top_from_partner
    )

    one_pass = replace(view, consecutive_passes=1)
    two_passes = replace(view, consecutive_passes=2)
    assert encode_policy_observation(one_pass) != encode_policy_observation(two_passes)

    continue_after_bombs = replace(
        view,
        bomb_queue=(1, 2, 3),
        bomb_resume_player=1,
        bomb_close_behavior="continue",
    )
    resolve_after_bombs = replace(
        view,
        bomb_queue=(1, 2, 3),
        bomb_resume_player=1,
        bomb_close_behavior="resolve_trick",
    )
    assert encode_policy_observation(continue_after_bombs) != encode_policy_observation(
        resolve_after_bombs
    )

    captured_by_self = replace(
        view,
        captured_card_ids=((0,), (), (), ()),
        captured_points=(5, 0, 0, 0),
    )
    captured_by_opponent = replace(
        view,
        captured_card_ids=((), (0,), (), ()),
        captured_points=(0, 5, 0, 0),
    )
    assert encode_policy_observation(captured_by_self) != encode_policy_observation(
        captured_by_opponent
    )


def test_private_exchange_provenance_changes_the_encoding() -> None:
    round_ = TichuRound.new(seed=0, dealer=0, config=_play_only_config())
    view = policy_view(round_.observation(round_.current_player))

    from_left = replace(
        view,
        private_exchange=PolicyExchange(
            sent=(),
            received=(PolicyExchangeCard(player=1, card_id=0),),
        ),
    )
    from_partner = replace(
        view,
        private_exchange=PolicyExchange(
            sent=(),
            received=(PolicyExchangeCard(player=2, card_id=0),),
        ),
    )
    assert encode_policy_observation(from_left) != encode_policy_observation(from_partner)


def test_action_encoding_is_invariant_to_absolute_seat_rotation() -> None:
    observer_zero = ExchangeAction(((1, 10), (2, 20), (3, 30)))
    observer_two = ExchangeAction(((0, 20), (1, 30), (3, 10)))

    assert encode_action(observer_zero, observer=0) == encode_action(
        observer_two,
        observer=2,
    )
    assert encode_action(DragonGiftAction(1), observer=0) == encode_action(
        DragonGiftAction(3),
        observer=2,
    )


def test_round_observation_tracks_public_play_state() -> None:
    round_ = TichuRound.new(seed=4, dealer=0, config=_play_only_config())
    player = round_.current_player
    assert player is not None
    action = next(
        action
        for action in round_.legal_actions()
        if isinstance(action, PlayAction)
    )

    round_ = round_.apply(action)
    observation = round_.observation(round_.current_player)

    assert observation["top_player"] == player
    assert observation["current_trick_card_ids"] == list(action.combination.card_ids)
    assert observation["played_card_ids"][player] == list(action.combination.card_ids)
    assert observation["consecutive_passes"] == 0


def test_exchange_observation_reveals_only_the_players_private_transfers() -> None:
    round_ = TichuRound.new(seed=5, dealer=0)
    while round_.phase is Phase.GRAND_TICHU:
        round_ = round_.apply(GrandTichuDecision(False))

    submitted: dict[int, ExchangeAction] = {}
    while round_.phase is Phase.EXCHANGE:
        player = round_.current_player
        assert player is not None
        assert round_.observation(player)["private_exchange"] is None
        action = round_.legal_actions()[0]
        assert isinstance(action, ExchangeAction)
        submitted[player] = action
        round_ = round_.apply(action)

    for player in range(4):
        private = round_.observation(player)["private_exchange"]
        assert private is not None

        expected_sent = {
            (recipient, card_id)
            for recipient, card_id in submitted[player].transfers
        }
        expected_received = {
            (sender, card_id)
            for sender, exchange in submitted.items()
            for recipient, card_id in exchange.transfers
            if recipient == player
        }
        assert {
            (transfer["player"], transfer["card_id"])
            for transfer in private["sent"]
        } == expected_sent
        assert {
            (transfer["player"], transfer["card_id"])
            for transfer in private["received"]
        } == expected_received
        assert len(private["sent"]) == 3
        assert len(private["received"]) == 3
