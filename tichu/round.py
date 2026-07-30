from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import permutations
from typing import Any, Iterable, Sequence, cast

from .actions import (
    Action,
    BombAction,
    DeclareSmallTichu,
    DragonGiftAction,
    ExchangeAction,
    GrandTichuDecision,
    PassAction,
    PlayAction,
    action_from_dict,
    action_to_dict,
    combination_to_dict,
)
from .cards import CARD_BY_ID, DECK, Special
from .config import TichuRoundConfig
from .combinations import (
    Combination,
    CombinationKind,
    generate_combinations,
    resolve_for_context,
)
from .observations import (
    BombCloseBehaviorName,
    CombinationObservation,
    ExchangeCardObservation,
    PhaseName,
    PrivateExchangeObservation,
    RoundObservation,
)

PlayerCards = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
PlayerFlags = tuple[bool, bool, bool, bool]
PlayerCalls = tuple[int | None, int | None, int | None, int | None]
PendingExchanges = tuple[
    ExchangeAction | None,
    ExchangeAction | None,
    ExchangeAction | None,
    ExchangeAction | None,
]
PublicEvent = tuple[tuple[str, Any], ...]


class Phase(str, Enum):
    GRAND_TICHU = "grand_tichu"
    EXCHANGE = "exchange"
    PLAY = "play"
    BOMB_WINDOW = "bomb_window"
    DRAGON_GIFT = "dragon_gift"
    ROUND_OVER = "round_over"


class BombCloseBehavior(str, Enum):
    CONTINUE = "continue"
    RESOLVE_TRICK = "resolve_trick"


def _phase_name(phase: Phase) -> PhaseName:
    names: dict[Phase, PhaseName] = {
        Phase.GRAND_TICHU: "grand_tichu",
        Phase.EXCHANGE: "exchange",
        Phase.PLAY: "play",
        Phase.BOMB_WINDOW: "bomb_window",
        Phase.DRAGON_GIFT: "dragon_gift",
        Phase.ROUND_OVER: "round_over",
    }
    return names[phase]


@dataclass(frozen=True, slots=True)
class ActionRecord:
    player: int
    phase: Phase
    action: Action

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "phase": self.phase.value,
            "action": action_to_dict(self.action),
        }


def team_of(player: int) -> int:
    return player % 2


def partner_of(player: int) -> int:
    return (player + 2) % 4


def _deck_for_config(config: TichuRoundConfig) -> tuple[int, ...]:
    return tuple(
        card.card_id
        for card in DECK
        if config.special_cards or card.special is None
    )


def _player_cards(
    cards: Mapping[int, Iterable[int]] | Sequence[Iterable[int]],
    *,
    sort_cards: bool = False,
) -> PlayerCards:
    if isinstance(cards, Mapping):
        values = [tuple(cards.get(player, ())) for player in range(4)]
    else:
        if len(cards) != 4:
            raise ValueError("Player card zones must contain exactly four entries")
        values = [tuple(zone) for zone in cards]
    if sort_cards:
        values = [tuple(sorted(zone)) for zone in values]
    return values[0], values[1], values[2], values[3]


def _player_flags(flags: Iterable[bool]) -> PlayerFlags:
    values = tuple(bool(value) for value in flags)
    if len(values) != 4:
        raise ValueError("Player flags must contain exactly four entries")
    return values  # type: ignore[return-value]


def _player_calls(calls: Mapping[int, int | None] | Sequence[int | None]) -> PlayerCalls:
    if isinstance(calls, Mapping):
        values = tuple(calls.get(player) for player in range(4))
    else:
        values = tuple(calls)
    if len(values) != 4:
        raise ValueError("Tichu calls must contain exactly four entries")
    return values  # type: ignore[return-value]


def _pending_exchanges(
    exchanges: Mapping[int, ExchangeAction] | Sequence[ExchangeAction | None],
) -> PendingExchanges:
    if isinstance(exchanges, Mapping):
        values = tuple(exchanges.get(player) for player in range(4))
    else:
        values = tuple(exchanges)
    if len(values) != 4:
        raise ValueError("Pending exchanges must contain exactly four entries")
    return values  # type: ignore[return-value]


def _set_player_cards(
    cards: PlayerCards,
    player: int,
    values: Iterable[int],
    *,
    sort_cards: bool = False,
) -> PlayerCards:
    zones = list(cards)
    zone = tuple(values)
    zones[player] = tuple(sorted(zone)) if sort_cards else zone
    return zones[0], zones[1], zones[2], zones[3]


def _add_to_player_cards(cards: PlayerCards, player: int, values: Iterable[int]) -> PlayerCards:
    return _set_player_cards(cards, player, cards[player] + tuple(values))


def _set_player_call(calls: PlayerCalls, player: int, value: int | None) -> PlayerCalls:
    updated = list(calls)
    updated[player] = value
    return updated[0], updated[1], updated[2], updated[3]


def _set_player_flag(flags: PlayerFlags, player: int, value: bool) -> PlayerFlags:
    updated = list(flags)
    updated[player] = value
    return updated[0], updated[1], updated[2], updated[3]


def _set_pending_exchange(
    exchanges: PendingExchanges,
    player: int,
    action: ExchangeAction | None,
) -> PendingExchanges:
    updated = list(exchanges)
    updated[player] = action
    return updated[0], updated[1], updated[2], updated[3]


def _deal_to_players(
    deck: Iterable[int],
    hands: PlayerCards,
    dealer: int,
    *,
    count: int | None = None,
) -> tuple[PlayerCards, tuple[int, ...]]:
    remaining = list(deck)
    dealt_hands = [list(hand) for hand in hands]
    cursor = dealer
    dealt = 0
    while remaining and (count is None or dealt < count):
        dealt_hands[cursor].append(remaining.pop(0))
        cursor = (cursor + 1) % 4
        dealt += 1
    if count is not None and dealt != count:
        raise AssertionError("The configured deck is too small for the initial deal")
    return _player_cards(dealt_hands, sort_cards=True), tuple(remaining)


def _opening_player(hands: PlayerCards, dealer: int) -> int:
    mahjong_holder = next(
        (
            player
            for player in range(4)
            if any(CARD_BY_ID[card_id].special is Special.MAHJONG for card_id in hands[player])
        ),
        None,
    )
    return dealer if mahjong_holder is None else mahjong_holder


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze_json(raw)) for key, raw in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(raw) for raw in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw_json(raw) for key, raw in value}
        return [_thaw_json(raw) for raw in value]
    return value


def _event(**values: Any) -> PublicEvent:
    return tuple((key, _freeze_json(value)) for key, value in values.items())


def _event_to_dict(event: PublicEvent) -> dict[str, Any]:
    return {key: _thaw_json(value) for key, value in event}


def _combination_observation(combo: Combination) -> CombinationObservation:
    return {
        "card_ids": list(combo.card_ids),
        "kind": combo.kind.value,
        "primary_rank": combo.primary_rank,
        "length": combo.length,
        "phoenix_as": combo.phoenix_as,
    }


@dataclass(frozen=True, slots=True)
class TichuRound:
    seed: int
    dealer: int
    hands: PlayerCards
    deck_remaining: tuple[int, ...]
    phase: Phase = Phase.GRAND_TICHU
    current_player: int | None = 0
    config: TichuRoundConfig = field(default_factory=TichuRoundConfig)

    captured: PlayerCards = field(default_factory=lambda: ((), (), (), ()))
    discarded: tuple[int, ...] = ()
    pending_exchanges: PendingExchanges = field(default_factory=lambda: (None, None, None, None))
    tichu_calls: PlayerCalls = field(default_factory=lambda: (None, None, None, None))
    has_played_card: PlayerFlags = field(default_factory=lambda: (False, False, False, False))

    wish_rank: int | None = None
    top_combination: Combination | None = None
    trick_cards: tuple[int, ...] = ()
    last_play_player: int | None = None
    consecutive_passes: int = 0
    finish_order: tuple[int, ...] = ()

    bomb_queue: tuple[int, ...] = ()
    bomb_resume_player: int | None = None
    bomb_close_behavior: BombCloseBehavior | None = None

    pending_dragon_winner: int | None = None
    finalize_after_dragon: bool = False

    round_scores: tuple[int, int] | None = None
    action_records: tuple[ActionRecord, ...] = ()
    public_events: tuple[PublicEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "hands", _player_cards(self.hands, sort_cards=True))
        object.__setattr__(self, "captured", _player_cards(self.captured))
        object.__setattr__(self, "deck_remaining", tuple(self.deck_remaining))
        object.__setattr__(self, "discarded", tuple(self.discarded))
        object.__setattr__(self, "pending_exchanges", _pending_exchanges(self.pending_exchanges))
        object.__setattr__(self, "tichu_calls", _player_calls(self.tichu_calls))
        object.__setattr__(self, "has_played_card", _player_flags(self.has_played_card))
        object.__setattr__(self, "trick_cards", tuple(self.trick_cards))
        object.__setattr__(self, "finish_order", tuple(self.finish_order))
        object.__setattr__(self, "bomb_queue", tuple(self.bomb_queue))
        object.__setattr__(self, "action_records", tuple(self.action_records))
        object.__setattr__(self, "public_events", tuple(self.public_events))

    @classmethod
    def new(
        cls,
        seed: int = 0,
        dealer: int = 0,
        config: TichuRoundConfig | None = None,
    ) -> "TichuRound":
        config = TichuRoundConfig() if config is None else config
        if dealer not in range(4):
            raise ValueError("dealer must be 0, 1, 2, or 3")
        rng = random.Random(seed)
        deck = list(_deck_for_config(config))
        rng.shuffle(deck)
        hands: PlayerCards = ((), (), (), ())

        if config.grand_tichu:
            # The first 8 cards per player are taken one at a time. Grand Tichu
            # is decided before the ninth card is received.
            hands, deck_remaining = _deal_to_players(deck, hands, dealer, count=32)
        else:
            hands, deck_remaining = _deal_to_players(deck, hands, dealer)

        round_ = cls(
            seed=seed,
            dealer=dealer,
            hands=hands,
            deck_remaining=deck_remaining,
            config=config,
            phase=Phase.GRAND_TICHU if config.grand_tichu else Phase.PLAY,
            current_player=dealer,
        )
        if not config.grand_tichu:
            round_ = round_._start_after_deal()
        round_.assert_valid()
        return round_

    @classmethod
    def replay(
        cls,
        seed: int,
        actions: Iterable[dict[str, Any]],
        dealer: int = 0,
        config: TichuRoundConfig | None = None,
    ) -> "TichuRound":
        round_ = cls.new(seed=seed, dealer=dealer, config=config)
        for index, raw_action in enumerate(actions):
            try:
                action_data = raw_action["action"] if "action" in raw_action else raw_action
                round_ = round_.apply(action_from_dict(action_data))
            except Exception as exc:  # pragma: no cover - convenience error path
                raise ValueError(f"Replay failed at action {index}: {raw_action}") from exc
        return round_

    @property
    def terminal(self) -> bool:
        return self.phase is Phase.ROUND_OVER

    @property
    def action_log(self) -> tuple[dict[str, Any], ...]:
        return tuple(record.to_dict() for record in self.action_records)

    @property
    def public_history(self) -> tuple[dict[str, Any], ...]:
        return tuple(_event_to_dict(event) for event in self.public_events)

    def legal_actions(self) -> tuple[Action, ...]:
        if self.current_player is None or self.terminal:
            return ()
        player = self.current_player

        if self.phase is Phase.GRAND_TICHU:
            if not self.config.grand_tichu:
                return ()
            return (GrandTichuDecision(False), GrandTichuDecision(True))

        if self.phase is Phase.EXCHANGE:
            if not self.config.card_exchange:
                return ()
            recipients = tuple(other for other in range(4) if other != player)
            return tuple(
                ExchangeAction(tuple(sorted(zip(recipients, selected), key=lambda x: x[0])))
                for selected in permutations(self.hands[player], 3)
            )

        if self.phase is Phase.DRAGON_GIFT:
            winner = self.pending_dragon_winner
            assert winner is not None
            return tuple(
                DragonGiftAction(recipient)
                for recipient in range(4)
                if team_of(recipient) != team_of(winner)
            )

        if self.phase is Phase.BOMB_WINDOW:
            if not self.config.out_of_turn_bombs:
                return ()
            bombs = self._legal_bombs(player)
            return (BombAction(None),) + tuple(BombAction(combo) for combo in bombs)

        if self.phase is not Phase.PLAY:
            return ()

        actions: list[Action] = []
        if (
            self.config.small_tichu
            and not self.has_played_card[player]
            and self.tichu_calls[player] is None
        ):
            actions.append(DeclareSmallTichu())

        candidate_plays: list[PlayAction] = []
        for raw_combo in self._combinations_in_hand(player):
            if raw_combo.is_dog:
                if self.top_combination is None:
                    candidate_plays.append(PlayAction(raw_combo))
                continue

            try:
                combo = resolve_for_context(raw_combo, self.top_combination)
            except ValueError:
                continue

            if self.top_combination is not None and not combo.beats(self.top_combination):
                continue

            if combo.contains_special(Special.MAHJONG):
                candidate_plays.append(PlayAction(combo, None))
                if self.config.wishes:
                    candidate_plays.extend(PlayAction(combo, rank) for rank in range(2, 15))
            else:
                candidate_plays.append(PlayAction(combo))

        forced = []
        if self.config.wishes and self.wish_rank is not None:
            forced = [
                action
                for action in candidate_plays
                if action.combination.contains_normal_rank(self.wish_rank)
            ]
        if forced:
            candidate_plays = forced
        elif self.top_combination is not None:
            actions.append(PassAction())

        actions.extend(candidate_plays)
        return tuple(actions)

    def apply(
        self,
        action: Action,
        *,
        legal_actions: Sequence[Action] | None = None,
    ) -> "TichuRound":
        if self.current_player is None or self.terminal:
            raise ValueError(f"No legal actions are available in {self.phase.value}")
        actions = self.legal_actions() if legal_actions is None else legal_actions
        if action not in actions:
            raise ValueError(
                f"Illegal action for player {self.current_player} in {self.phase.value}: {action}"
            )
        player = self.current_player
        assert player is not None

        state = replace(
            self,
            action_records=self.action_records + (ActionRecord(player, self.phase, action),),
        )

        if isinstance(action, GrandTichuDecision):
            state = state._apply_grand_tichu(player, action)
        elif isinstance(action, ExchangeAction):
            state = state._apply_exchange(player, action)
        elif isinstance(action, DeclareSmallTichu):
            state = state._with_tichu_call(player, 100)._with_event(
                type="small_tichu",
                player=player,
            )
        elif isinstance(action, PlayAction):
            state = state._apply_play(player, action)
        elif isinstance(action, PassAction):
            state = state._apply_pass(player)
        elif isinstance(action, BombAction):
            state = state._apply_bomb(player, action)
        elif isinstance(action, DragonGiftAction):
            state = state._apply_dragon_gift(player, action)
        else:  # pragma: no cover
            raise TypeError(type(action))

        state.assert_valid()
        return state

    def _with_event(self, **values: Any) -> "TichuRound":
        return replace(self, public_events=self.public_events + (_event(**values),))

    def _with_hand(self, player: int, cards: Iterable[int]) -> "TichuRound":
        return replace(self, hands=_set_player_cards(self.hands, player, cards, sort_cards=True))

    def _with_captured(self, player: int, cards: Iterable[int]) -> "TichuRound":
        return replace(self, captured=_set_player_cards(self.captured, player, cards))

    def _with_tichu_call(self, player: int, value: int | None) -> "TichuRound":
        return replace(self, tichu_calls=_set_player_call(self.tichu_calls, player, value))

    def _with_has_played_card(self, player: int, value: bool) -> "TichuRound":
        return replace(self, has_played_card=_set_player_flag(self.has_played_card, player, value))

    def _remove_from_hand(self, player: int, card_ids: Iterable[int]) -> "TichuRound":
        hand = list(self.hands[player])
        for card_id in card_ids:
            hand.remove(card_id)
        return self._with_hand(player, hand)

    def _add_captured(self, player: int, card_ids: Iterable[int]) -> "TichuRound":
        return replace(self, captured=_add_to_player_cards(self.captured, player, card_ids))

    def _apply_grand_tichu(self, player: int, action: GrandTichuDecision) -> "TichuRound":
        state = self
        if action.call:
            state = state._with_tichu_call(player, 200)._with_event(
                type="grand_tichu",
                player=player,
            )

        next_player = (player + 1) % 4
        if next_player != state.dealer:
            return replace(state, current_player=next_player)

        hands, deck_remaining = _deal_to_players(state.deck_remaining, state.hands, state.dealer)
        return replace(state, hands=hands, deck_remaining=deck_remaining)._start_after_deal()

    def _start_after_deal(self) -> "TichuRound":
        if self.config.card_exchange:
            return replace(self, phase=Phase.EXCHANGE, current_player=self.dealer)
        return self._start_play()

    def _start_play(self) -> "TichuRound":
        return replace(
            self,
            phase=Phase.PLAY,
            current_player=_opening_player(self.hands, self.dealer),
        )

    def _apply_exchange(self, player: int, action: ExchangeAction) -> "TichuRound":
        state = replace(
            self,
            pending_exchanges=_set_pending_exchange(self.pending_exchanges, player, action),
        )._with_event(type="exchange_submitted", player=player)

        next_player = (player + 1) % 4
        if next_player != state.dealer:
            return replace(state, current_player=next_player)

        if any(exchange is None for exchange in state.pending_exchanges):
            raise AssertionError("Cannot complete exchange before all players submit")

        outgoing: dict[int, list[int]] = {player_id: [] for player_id in range(4)}
        incoming: dict[int, list[int]] = {player_id: [] for player_id in range(4)}
        for sender, exchange in enumerate(state.pending_exchanges):
            assert exchange is not None
            for recipient, card_id in exchange.transfers:
                outgoing[sender].append(card_id)
                incoming[recipient].append(card_id)

        hands = [list(hand) for hand in state.hands]
        for sender in range(4):
            for card_id in outgoing[sender]:
                hands[sender].remove(card_id)
        for recipient in range(4):
            hands[recipient].extend(incoming[recipient])

        return replace(
            state,
            hands=_player_cards(hands, sort_cards=True),
            pending_exchanges=(None, None, None, None),
        )._with_event(type="exchange_complete")._start_play()

    def _apply_play(self, player: int, action: PlayAction) -> "TichuRound":
        combo = action.combination
        state = self._remove_from_hand(player, combo.card_ids)._with_has_played_card(player, True)

        old_wish = state.wish_rank
        if state.config.wishes and old_wish is not None and combo.contains_normal_rank(old_wish):
            state = replace(state, wish_rank=None)._with_event(
                type="wish_fulfilled",
                rank=old_wish,
                player=player,
            )

        if state.config.wishes and combo.contains_special(Special.MAHJONG):
            state = replace(state, wish_rank=action.wish_rank)
            if action.wish_rank is not None:
                state = state._with_event(
                    type="wish_made",
                    rank=action.wish_rank,
                    player=player,
                )

        state = state._with_event(
            type="play",
            player=player,
            combination=combination_to_dict(combo),
            wish_rank=action.wish_rank,
        )

        if combo.is_dog:
            state = replace(state, discarded=state.discarded + combo.card_ids)
            if not state.hands[player]:
                state = state._record_finish(player)
                state, ended = state._finish_condition_after_play()
                if ended:
                    return state
            target = partner_of(player)
            if not state.hands[target]:
                target = state._next_active(target)
            return replace(
                state._reset_trick(),
                phase=Phase.PLAY,
                current_player=target,
            )

        state = replace(
            state,
            trick_cards=state.trick_cards + combo.card_ids,
            top_combination=combo,
            last_play_player=player,
            consecutive_passes=0,
        )

        if not state.hands[player]:
            state = state._record_finish(player)
            state, ended = state._finish_condition_after_play()
            if ended:
                return state

        resume_player = state._next_active(player)
        if state.config.out_of_turn_bombs:
            close_behavior = (
                BombCloseBehavior.RESOLVE_TRICK
                if combo.is_bomb
                else BombCloseBehavior.CONTINUE
            )
            return state._open_bomb_window(
                anchor=player,
                resume_player=resume_player,
                close_behavior=close_behavior,
            )
        return replace(
            state,
            phase=Phase.PLAY,
            current_player=resume_player,
        )

    def _apply_pass(self, player: int) -> "TichuRound":
        state = self._with_event(type="pass", player=player)
        state = replace(state, consecutive_passes=state.consecutive_passes + 1)
        needed = state._passes_needed()
        should_resolve = state.consecutive_passes >= needed
        resume_player = state._next_active(player)
        if state.config.out_of_turn_bombs:
            return state._open_bomb_window(
                anchor=player,
                resume_player=resume_player,
                close_behavior=(
                    BombCloseBehavior.RESOLVE_TRICK
                    if should_resolve
                    else BombCloseBehavior.CONTINUE
                ),
            )
        if should_resolve:
            return state._resolve_trick()
        return replace(
            state,
            phase=Phase.PLAY,
            current_player=resume_player,
        )

    def _apply_bomb(self, player: int, action: BombAction) -> "TichuRound":
        if action.combination is None:
            state = self._with_event(type="decline_bomb", player=player)
            assert state.bomb_queue and state.bomb_queue[0] == player
            queue = state.bomb_queue[1:]
            state = replace(state, bomb_queue=queue)
            if queue:
                return replace(state, current_player=queue[0])
            return state._close_bomb_window()

        combo = action.combination
        state = self._remove_from_hand(player, combo.card_ids)._with_has_played_card(player, True)
        state = replace(
            state,
            trick_cards=state.trick_cards + combo.card_ids,
            top_combination=combo,
            last_play_player=player,
            consecutive_passes=0,
        )._with_event(
            type="bomb",
            player=player,
            combination=combination_to_dict(combo),
        )

        if state.config.wishes and state.wish_rank is not None and combo.contains_normal_rank(state.wish_rank):
            old_wish = state.wish_rank
            state = replace(state, wish_rank=None)._with_event(
                type="wish_fulfilled",
                rank=old_wish,
                player=player,
            )

        if not state.hands[player]:
            state = state._record_finish(player)
            state, ended = state._finish_condition_after_play()
            if ended:
                return state

        return state._open_bomb_window(
            anchor=player,
            resume_player=state._next_active(player),
            close_behavior=BombCloseBehavior.RESOLVE_TRICK,
        )

    def _apply_dragon_gift(self, player: int, action: DragonGiftAction) -> "TichuRound":
        if player != self.pending_dragon_winner:
            raise AssertionError("Only the Dragon trick winner chooses the recipient")
        state = self._add_captured(action.recipient, self.trick_cards)._with_event(
            type="dragon_gift",
            winner=player,
            recipient=action.recipient,
        )
        state = replace(state._reset_trick(), pending_dragon_winner=None)

        if state.finalize_after_dragon:
            return replace(state, finalize_after_dragon=False)._finalize_normal_round()

        return replace(
            state,
            phase=Phase.PLAY,
            current_player=player if state.hands[player] else state._next_active(player),
        )

    def _open_bomb_window(
        self,
        *,
        anchor: int,
        resume_player: int,
        close_behavior: BombCloseBehavior,
    ) -> "TichuRound":
        if not self.config.out_of_turn_bombs:
            raise AssertionError("Bomb windows require out_of_turn_bombs")
        if self.top_combination is None or self.top_combination.is_dog:
            return replace(self, phase=Phase.PLAY, current_player=resume_player)

        state = replace(
            self,
            bomb_resume_player=resume_player,
            bomb_close_behavior=close_behavior,
        )
        queue = tuple(state._active_cycle_after(anchor))
        if not any(state._legal_bombs(player) for player in queue):
            return state._close_bomb_window()

        return replace(
            state,
            phase=Phase.BOMB_WINDOW,
            bomb_queue=queue,
            current_player=queue[0],
        )

    def _close_bomb_window(self) -> "TichuRound":
        behavior = self.bomb_close_behavior
        resume = self.bomb_resume_player
        state = replace(
            self,
            bomb_queue=(),
            bomb_close_behavior=None,
            bomb_resume_player=None,
        )
        if behavior is BombCloseBehavior.RESOLVE_TRICK:
            return state._resolve_trick()
        return replace(state, phase=Phase.PLAY, current_player=resume)

    def _resolve_trick(self) -> "TichuRound":
        winner = self.last_play_player
        top = self.top_combination
        if winner is None or top is None:
            raise AssertionError("Cannot resolve an empty trick")

        if top.contains_special(Special.DRAGON):
            return replace(
                self,
                phase=Phase.DRAGON_GIFT,
                pending_dragon_winner=winner,
                current_player=winner,
            )

        return replace(
            self._add_captured(winner, self.trick_cards)._with_event(
                type="trick_won",
                player=winner,
            )._reset_trick(),
            phase=Phase.PLAY,
            current_player=winner if self.hands[winner] else self._next_active(winner),
        )

    def _record_finish(self, player: int) -> "TichuRound":
        if player in self.finish_order:
            return self
        position = len(self.finish_order) + 1
        return replace(
            self,
            finish_order=self.finish_order + (player,),
        )._with_event(type="went_out", player=player, position=position)

    def _finish_condition_after_play(self) -> tuple["TichuRound", bool]:
        if len(self.finish_order) >= 2 and team_of(self.finish_order[0]) == team_of(self.finish_order[1]):
            return self._finalize_double_victory(), True

        if len(self.finish_order) == 3:
            if (
                self.top_combination is not None
                and self.top_combination.contains_special(Special.DRAGON)
            ):
                return replace(
                    self,
                    phase=Phase.DRAGON_GIFT,
                    pending_dragon_winner=self.last_play_player,
                    current_player=self.last_play_player,
                    finalize_after_dragon=True,
                ), True

            state = self
            if state.trick_cards and state.last_play_player is not None:
                state = state._add_captured(state.last_play_player, state.trick_cards)._with_event(
                    type="final_trick_won",
                    player=state.last_play_player,
                )
            return state._reset_trick()._finalize_normal_round(), True
        return self, False

    def _finalize_double_victory(self) -> "TichuRound":
        winning_team = team_of(self.finish_order[0])
        scores = [0, 0]
        scores[winning_team] = 200
        self._apply_tichu_scores(scores)
        state = replace(
            self,
            round_scores=(scores[0], scores[1]),
            phase=Phase.ROUND_OVER,
            current_player=None,
            bomb_queue=(),
        )
        return state._with_event(
            type="round_over",
            double_victory=True,
            scores=list(state.round_scores or ()),
        )

    def _finalize_normal_round(self) -> "TichuRound":
        if len(self.finish_order) != 3:
            raise AssertionError("Normal round scoring requires exactly three players out")
        tailender = next(player for player in range(4) if self.hands[player])
        first_out = self.finish_order[0]

        opponent_recipient = (tailender + 1) % 4
        state = self._add_captured(opponent_recipient, self.hands[tailender])
        state = state._with_hand(tailender, ())
        state = state._add_captured(first_out, state.captured[tailender])
        state = state._with_captured(tailender, ())

        scores = [0, 0]
        for player in range(4):
            scores[team_of(player)] += sum(CARD_BY_ID[card_id].points for card_id in state.captured[player])
        state._apply_tichu_scores(scores)

        state = replace(
            state,
            round_scores=(scores[0], scores[1]),
            phase=Phase.ROUND_OVER,
            current_player=None,
            bomb_queue=(),
        )
        return state._with_event(
            type="round_over",
            double_victory=False,
            scores=list(state.round_scores or ()),
        )

    def _apply_tichu_scores(self, scores: list[int]) -> None:
        if not self.finish_order:
            return
        winner = self.finish_order[0]
        for player, value in enumerate(self.tichu_calls):
            if value is None:
                continue
            scores[team_of(player)] += value if player == winner else -value

    def _combinations_in_hand(self, player: int) -> tuple[Combination, ...]:
        return generate_combinations(self.hands[player], self.config.bombs)

    def _legal_bombs(self, player: int) -> tuple[Combination, ...]:
        if not self.config.out_of_turn_bombs or self.top_combination is None:
            return ()
        return tuple(
            combo
            for combo in self._combinations_in_hand(player)
            if combo.is_bomb and combo.beats(self.top_combination)
        )

    def _passes_needed(self) -> int:
        if self.last_play_player is None:
            raise AssertionError("No last player for pass counting")
        return sum(
            1
            for player in range(4)
            if player != self.last_play_player and bool(self.hands[player])
        )

    def _next_active(self, player: int) -> int:
        for offset in range(1, 5):
            candidate = (player + offset) % 4
            if self.hands[candidate]:
                return candidate
        raise RuntimeError("No active players remain")

    def _active_cycle_after(self, player: int) -> tuple[int, ...]:
        return tuple(
            candidate
            for offset in range(1, 5)
            if self.hands[candidate := (player + offset) % 4]
        )

    def _reset_trick(self) -> "TichuRound":
        return replace(
            self,
            top_combination=None,
            trick_cards=(),
            last_play_player=None,
            consecutive_passes=0,
        )

    def _played_cards_by_player(self) -> PlayerCards:
        played: list[list[int]] = [[], [], [], []]
        for record in self.action_records:
            action = record.action
            if isinstance(action, PlayAction):
                played[record.player].extend(action.combination.card_ids)
            elif isinstance(action, BombAction) and action.combination is not None:
                played[record.player].extend(action.combination.card_ids)
        return _player_cards(played)

    def _private_exchange_observation(
        self,
        player: int,
    ) -> PrivateExchangeObservation | None:
        if not self.config.card_exchange or self.phase is Phase.EXCHANGE:
            return None

        exchanges = {
            record.player: record.action
            for record in self.action_records
            if isinstance(record.action, ExchangeAction)
        }
        if len(exchanges) != 4:
            return None

        own_exchange = exchanges[player]
        sent = [
            ExchangeCardObservation(player=recipient, card_id=card_id)
            for recipient, card_id in own_exchange.transfers
        ]
        received = sorted(
            (
                ExchangeCardObservation(player=sender, card_id=card_id)
                for sender, exchange in exchanges.items()
                for recipient, card_id in exchange.transfers
                if recipient == player
            ),
            key=lambda transfer: transfer["player"],
        )
        if len(sent) != 3 or len(received) != 3:
            raise AssertionError("A completed exchange must send and receive three cards")
        return PrivateExchangeObservation(sent=sent, received=received)

    def observation(self, player: int) -> RoundObservation:
        if player not in range(4):
            raise ValueError("player must be 0, 1, 2, or 3")
        played_cards = self._played_cards_by_player()
        return RoundObservation(
            config=self.config.to_dict(),
            phase=_phase_name(self.phase),
            current_player=self.current_player,
            player=player,
            team=team_of(player),
            dealer=self.dealer,
            hand=[
                {"card_id": card_id, "label": CARD_BY_ID[card_id].label}
                for card_id in sorted(self.hands[player])
            ],
            hand_sizes=[len(self.hands[p]) for p in range(4)],
            played_card_ids=[list(played_cards[p]) for p in range(4)],
            captured_card_ids=[list(self.captured[p]) for p in range(4)],
            captured_points=[
                sum(CARD_BY_ID[card_id].points for card_id in self.captured[p])
                for p in range(4)
            ],
            current_trick_card_ids=list(self.trick_cards),
            top_combination=(
                None if self.top_combination is None else _combination_observation(self.top_combination)
            ),
            top_player=self.last_play_player,
            consecutive_passes=self.consecutive_passes,
            bomb_queue=list(self.bomb_queue),
            bomb_resume_player=self.bomb_resume_player,
            bomb_close_behavior=(
                None
                if self.bomb_close_behavior is None
                else cast(BombCloseBehaviorName, self.bomb_close_behavior.value)
            ),
            wish_rank=self.wish_rank,
            finish_order=list(self.finish_order),
            tichu_calls={
                player_id: value
                for player_id, value in enumerate(self.tichu_calls)
                if value is not None
            },
            has_played_card=list(self.has_played_card),
            private_exchange=self._private_exchange_observation(player),
            public_history=list(self.public_history),
            round_scores=self.round_scores,
        )

    def full_state(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "dealer": self.dealer,
            "config": self.config.to_dict(),
            "phase": self.phase.value,
            "current_player": self.current_player,
            "hands": {str(player): list(cards) for player, cards in enumerate(self.hands)},
            "captured": {str(player): list(cards) for player, cards in enumerate(self.captured)},
            "discarded": list(self.discarded),
            "wish_rank": self.wish_rank,
            "top_combination": (
                None if self.top_combination is None else combination_to_dict(self.top_combination)
            ),
            "trick_cards": list(self.trick_cards),
            "finish_order": list(self.finish_order),
            "tichu_calls": {
                str(player): value
                for player, value in enumerate(self.tichu_calls)
                if value is not None
            },
            "round_scores": None if self.round_scores is None else list(self.round_scores),
        }

    def save_replay(self, path: str) -> None:
        payload = {
            "seed": self.seed,
            "dealer": self.dealer,
            "config": self.config.to_dict(),
            "actions": list(self.action_log),
            "final_state": self.full_state(),
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)

    def assert_valid(self) -> None:
        zones: list[int] = []
        for hand in self.hands:
            if len(hand) != len(set(hand)):
                raise AssertionError("A hand contains a duplicate physical card")
            zones.extend(hand)
        for pile in self.captured:
            zones.extend(pile)
        zones.extend(self.discarded)
        zones.extend(self.trick_cards)
        zones.extend(self.deck_remaining)

        if len(zones) != len(set(zones)):
            duplicates = [card_id for card_id in set(zones) if zones.count(card_id) > 1]
            raise AssertionError(f"Cards appear in multiple zones: {duplicates}")
        expected_cards = set(_deck_for_config(self.config))
        if set(zones) != expected_cards:
            missing = expected_cards - set(zones)
            extra = set(zones) - expected_cards
            raise AssertionError(f"Card conservation failed; missing={missing}, extra={extra}")

        if self.current_player is not None and self.current_player not in range(4):
            raise AssertionError("Invalid current player")
        if self.phase is Phase.GRAND_TICHU and not self.config.grand_tichu:
            raise AssertionError("Grand Tichu phase is disabled")
        if self.phase is Phase.EXCHANGE and not self.config.card_exchange:
            raise AssertionError("Card exchange phase is disabled")
        if self.phase is Phase.BOMB_WINDOW and not self.config.out_of_turn_bombs:
            raise AssertionError("Out-of-turn bomb windows are disabled")
        if not self.config.out_of_turn_bombs and (
            self.bomb_queue
            or self.bomb_resume_player is not None
            or self.bomb_close_behavior is not None
        ):
            raise AssertionError("Disabled out-of-turn bombs cannot retain bomb-window state")
        if self.phase is Phase.DRAGON_GIFT and not self.config.special_cards:
            raise AssertionError("Dragon gift phase requires special cards")
        if self.wish_rank is not None and self.wish_rank not in range(2, 15):
            raise AssertionError("Invalid Mahjong wish")
        if self.wish_rank is not None and not self.config.wishes:
            raise AssertionError("Mahjong wishes are disabled")
        for player, value in enumerate(self.tichu_calls):
            if value is None:
                continue
            if value == 100 and not self.config.small_tichu:
                raise AssertionError("Small Tichu calls are disabled")
            if value == 200 and not self.config.grand_tichu:
                raise AssertionError("Grand Tichu calls are disabled")
            if value not in (100, 200):
                raise AssertionError("Invalid Tichu call value")
        if len(set(self.finish_order)) != len(self.finish_order):
            raise AssertionError("A player appears twice in finish order")
        for player in self.finish_order:
            if self.hands[player]:
                raise AssertionError("A finished player still has cards")
        if self.phase is Phase.ROUND_OVER and self.round_scores is None:
            raise AssertionError("A completed round must have scores")
