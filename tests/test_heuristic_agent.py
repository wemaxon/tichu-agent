from __future__ import annotations

from agents import available_agent_names, create_agent
from agents.heuristic import HeuristicAgent
from tichu.actions import (
    BombAction,
    DeclareSmallTichu,
    DragonGiftAction,
    ExchangeAction,
    GrandTichuDecision,
    PassAction,
    PlayAction,
)
from tichu.cards import DECK, Special, Suit
from tichu.combinations import Combination, CombinationKind
from tichu.config import TichuRoundConfig
from tichu.observations import CombinationObservation, PhaseName, RoundObservation
from tichu.round import TichuRound


def _normal(rank: int, suit: Suit = Suit.JADE) -> int:
    return next(card.card_id for card in DECK if card.rank == rank and card.suit is suit)


def _special(special: Special) -> int:
    return next(card.card_id for card in DECK if card.special is special)


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
    public_history: list[dict] | None = None,
) -> RoundObservation:
    return RoundObservation(
        config=TichuRoundConfig().to_dict(),
        phase=phase,
        current_player=current_player,
        player=current_player,
        team=current_player % 2,
        hand=[{"card_id": card_id, "label": str(card_id)} for card_id in hand],
        hand_sizes=list(hand_sizes or (len(hand), 0, 0, 0)),
        top_combination=(
            None if top_combination is None else _combination_observation(top_combination)
        ),
        wish_rank=None,
        finish_order=[],
        tichu_calls={},
        public_history=[] if public_history is None else public_history,
        round_scores=None,
    )


def test_heuristic_agent_is_registered() -> None:
    assert "heuristic" in available_agent_names()
    assert isinstance(create_agent("heuristic", seed=123), HeuristicAgent)


def test_heuristic_calls_grand_tichu_with_very_strong_opening_hand() -> None:
    hand = (
        _special(Special.DRAGON),
        _special(Special.PHOENIX),
        _normal(14, Suit.JADE),
        _normal(14, Suit.SWORD),
        _normal(13, Suit.JADE),
        _normal(13, Suit.SWORD),
        _normal(12, Suit.JADE),
        _normal(11, Suit.JADE),
    )

    action = HeuristicAgent().choose_action(
        _observation(phase="grand_tichu", hand=hand),
        (GrandTichuDecision(False), GrandTichuDecision(True)),
    )

    assert action == GrandTichuDecision(True)


def test_heuristic_declines_grand_tichu_with_weak_opening_hand() -> None:
    hand = (
        _normal(2, Suit.JADE),
        _normal(3, Suit.JADE),
        _normal(4, Suit.JADE),
        _normal(5, Suit.JADE),
        _normal(6, Suit.JADE),
        _normal(7, Suit.JADE),
        _normal(8, Suit.JADE),
        _normal(9, Suit.JADE),
    )

    action = HeuristicAgent().choose_action(
        _observation(phase="grand_tichu", hand=hand),
        (GrandTichuDecision(False), GrandTichuDecision(True)),
    )

    assert action == GrandTichuDecision(False)


def test_heuristic_exchanges_best_low_card_to_partner_and_low_cards_to_opponents() -> None:
    hand = (
        _normal(14, Suit.JADE),
        _normal(2, Suit.JADE),
        _normal(4, Suit.JADE),
        _normal(13, Suit.JADE),
        _normal(3, Suit.JADE),
    )
    expected = ExchangeAction(
        (
            (1, _normal(2, Suit.JADE)),
            (2, _normal(4, Suit.JADE)),
            (3, _normal(3, Suit.JADE)),
        )
    )

    action = HeuristicAgent().choose_action(
        _observation(phase="exchange", current_player=0, hand=hand),
        (expected,),
    )

    assert action == expected


def test_heuristic_passes_when_partner_is_winning_the_trick() -> None:
    top = _combo(CombinationKind.SINGLE, 10, (_normal(10, Suit.JADE),))
    jack = PlayAction(_combo(CombinationKind.SINGLE, 11, (_normal(11, Suit.JADE),)))

    action = HeuristicAgent().choose_action(
        _observation(
            current_player=0,
            top_combination=top,
            public_history=[{"type": "play", "player": 2}],
        ),
        (PassAction(), jack),
    )

    assert action == PassAction()


def test_heuristic_beats_opponent_with_cheapest_non_bomb() -> None:
    top = _combo(CombinationKind.SINGLE, 10, (_normal(10, Suit.JADE),))
    jack = PlayAction(_combo(CombinationKind.SINGLE, 11, (_normal(11, Suit.JADE),)))
    king = PlayAction(_combo(CombinationKind.SINGLE, 13, (_normal(13, Suit.JADE),)))

    action = HeuristicAgent().choose_action(
        _observation(
            current_player=0,
            top_combination=top,
            public_history=[{"type": "play", "player": 1}],
        ),
        (PassAction(), king, jack),
    )

    assert action == jack


def test_heuristic_declares_small_tichu_with_strong_full_hand() -> None:
    hand = (
        _special(Special.DRAGON),
        _special(Special.PHOENIX),
        _normal(14, Suit.JADE),
        _normal(14, Suit.SWORD),
        _normal(14, Suit.PAGODA),
        _normal(14, Suit.STAR),
        _normal(13, Suit.JADE),
        _normal(13, Suit.SWORD),
        _normal(13, Suit.PAGODA),
        _normal(13, Suit.STAR),
        _normal(11, Suit.JADE),
        _normal(10, Suit.JADE),
        _normal(9, Suit.JADE),
        _normal(8, Suit.JADE),
    )
    play = PlayAction(_combo(CombinationKind.SINGLE, 14, (_normal(14, Suit.JADE),)))

    action = HeuristicAgent().choose_action(
        _observation(hand=hand),
        (DeclareSmallTichu(), play),
    )

    assert action == DeclareSmallTichu()


def test_heuristic_uses_bomb_to_stop_nearly_out_opponent() -> None:
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

    action = HeuristicAgent().choose_action(
        _observation(
            phase="bomb_window",
            current_player=0,
            hand=bomb_combo.card_ids + (_normal(5, Suit.JADE),),
            hand_sizes=(5, 1, 4, 6),
            top_combination=_combo(CombinationKind.SINGLE, 14, (_normal(14, Suit.JADE),)),
            public_history=[{"type": "play", "player": 1}],
        ),
        (BombAction(None), BombAction(bomb_combo)),
    )

    assert action == BombAction(bomb_combo)


def test_heuristic_declines_bomb_when_partner_is_winning() -> None:
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

    action = HeuristicAgent().choose_action(
        _observation(
            phase="bomb_window",
            current_player=0,
            hand=bomb_combo.card_ids + (_normal(5, Suit.JADE),),
            hand_sizes=(5, 4, 1, 6),
            top_combination=_combo(CombinationKind.SINGLE, 14, (_normal(14, Suit.JADE),)),
            public_history=[{"type": "play", "player": 2}],
        ),
        (BombAction(None), BombAction(bomb_combo)),
    )

    assert action == BombAction(None)


def test_heuristic_round_chooses_only_legal_actions() -> None:
    round_ = TichuRound.new(seed=0)
    agents = [HeuristicAgent() for _ in range(4)]
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
