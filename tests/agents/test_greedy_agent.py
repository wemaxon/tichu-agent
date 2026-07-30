"""Tests for the deterministic greedy baseline."""

from __future__ import annotations

from agents import available_agent_names, create_agent
from agents.greedy import GreedyAgent
from tichu.actions import (
    BombAction,
    DragonGiftAction,
    ExchangeAction,
    GrandTichuDecision,
    PassAction,
    PlayAction,
)
from tichu.cards import DECK, Suit
from tichu.combinations import Combination, CombinationKind
from tichu.config import TichuRoundConfig
from tichu.observations import CombinationObservation, PhaseName, RoundObservation
from tichu.round import TichuRound


def _normal(rank: int, suit: Suit = Suit.JADE) -> int:
    return next(card.card_id for card in DECK if card.rank == rank and card.suit is suit)


def _combo(kind: CombinationKind, primary_rank: float, card_ids: tuple[int, ...]) -> Combination:
    return Combination(
        card_ids=tuple(sorted(card_ids)),
        kind=kind,
        primary_rank=primary_rank,
        length=len(card_ids),
    )


def _combination_observation(combo: Combination) -> CombinationObservation:
    return CombinationObservation(
        card_ids=list(combo.card_ids),
        kind=combo.kind.value,
        primary_rank=combo.primary_rank,
        length=combo.length,
        phoenix_as=combo.phoenix_as,
    )


def _observation(
    *,
    phase: PhaseName = "play",
    current_player: int = 0,
    hand: tuple[int, ...] = (),
    top_combination: Combination | None = None,
    hand_sizes: tuple[int, int, int, int] | None = None,
) -> RoundObservation:
    return RoundObservation(
        config=TichuRoundConfig(
            out_of_turn_bombs=phase == "bomb_window",
        ).to_dict(),
        phase=phase,
        current_player=current_player,
        player=current_player,
        team=current_player % 2,
        dealer=0,
        hand=[{"card_id": card_id, "label": str(card_id)} for card_id in hand],
        hand_sizes=list(hand_sizes or (len(hand), 0, 0, 0)),
        played_card_ids=[[], [], [], []],
        captured_card_ids=[[], [], [], []],
        captured_points=[0, 0, 0, 0],
        current_trick_card_ids=[],
        top_combination=(
            None if top_combination is None else _combination_observation(top_combination)
        ),
        top_player=None,
        consecutive_passes=0,
        bomb_queue=[],
        bomb_resume_player=None,
        bomb_close_behavior=None,
        wish_rank=None,
        finish_order=[],
        tichu_calls={},
        has_played_card=[False, False, False, False],
        private_exchange=None,
        public_history=[],
        round_scores=None,
    )


def test_greedy_agent_is_registered() -> None:
    assert "greedy" in available_agent_names()
    assert isinstance(create_agent("greedy", seed=123), GreedyAgent)


def test_greedy_declines_grand_tichu() -> None:
    actions = (GrandTichuDecision(False), GrandTichuDecision(True))
    action = GreedyAgent().choose_action(_observation(phase="grand_tichu"), actions)
    assert action == GrandTichuDecision(False)


def test_greedy_exchanges_lowest_cards() -> None:
    hand = (
        _normal(14),
        _normal(2),
        _normal(4),
        _normal(13),
        _normal(3),
    )
    expected = ExchangeAction(
        (
            (1, _normal(2)),
            (2, _normal(3)),
            (3, _normal(4)),
        )
    )
    actions = (expected,)

    action = GreedyAgent().choose_action(
        _observation(phase="exchange", hand=hand),
        actions,
    )

    assert action == expected


def test_greedy_follows_with_cheapest_non_bomb_play() -> None:
    top = _combo(CombinationKind.SINGLE, 10, (_normal(10),))
    jack = PlayAction(_combo(CombinationKind.SINGLE, 11, (_normal(11),)))
    king = PlayAction(_combo(CombinationKind.SINGLE, 13, (_normal(13),)))
    bomb = PlayAction(
        _combo(
            CombinationKind.FOUR_BOMB,
            2,
            (
                _normal(2, Suit.JADE),
                _normal(2, Suit.SWORD),
                _normal(2, Suit.PAGODA),
                _normal(2, Suit.STAR),
            ),
        )
    )

    action = GreedyAgent().choose_action(
        _observation(top_combination=top),
        (PassAction(), bomb, king, jack),
    )

    assert action == jack


def test_greedy_passes_instead_of_spending_bomb() -> None:
    bomb = BombAction(
        _combo(
            CombinationKind.FOUR_BOMB,
            2,
            (
                _normal(2, Suit.JADE),
                _normal(2, Suit.SWORD),
                _normal(2, Suit.PAGODA),
                _normal(2, Suit.STAR),
            ),
        )
    )

    action = GreedyAgent().choose_action(
        _observation(
            phase="bomb_window",
            hand=(
                _normal(2, Suit.JADE),
                _normal(2, Suit.SWORD),
                _normal(2, Suit.PAGODA),
                _normal(2, Suit.STAR),
                _normal(5),
            ),
        ),
        (BombAction(None), bomb),
    )

    assert action == BombAction(None)


def test_greedy_bombs_to_go_out() -> None:
    bomb_combo = _combo(
        CombinationKind.FOUR_BOMB,
        2,
        (
            _normal(2, Suit.JADE),
            _normal(2, Suit.SWORD),
            _normal(2, Suit.PAGODA),
            _normal(2, Suit.STAR),
        ),
    )

    action = GreedyAgent().choose_action(
        _observation(phase="bomb_window", hand=bomb_combo.card_ids),
        (BombAction(None), BombAction(bomb_combo)),
    )

    assert action == BombAction(bomb_combo)


def test_greedy_gives_dragon_to_opponent_with_most_cards() -> None:
    action = GreedyAgent().choose_action(
        _observation(
            phase="dragon_gift",
            hand_sizes=(5, 9, 5, 11),
        ),
        (DragonGiftAction(1), DragonGiftAction(3)),
    )

    assert action == DragonGiftAction(3)


def test_greedy_round_chooses_only_legal_actions() -> None:
    round_ = TichuRound.new(seed=0)
    agents = [GreedyAgent() for _ in range(4)]
    steps = 0

    while not round_.terminal:
        player = round_.current_player
        assert player is not None
        legal_actions = round_.legal_actions()
        action = agents[player].choose_action(round_.observation(player), legal_actions)
        assert action in legal_actions
        round_ = round_.apply(action)
        steps += 1
        assert steps <= 20_000

    assert round_.round_scores is not None
