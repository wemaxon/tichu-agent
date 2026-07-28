from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from itertools import permutations
from typing import Any, Iterable

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


def team_of(player: int) -> int:
    return player % 2


def partner_of(player: int) -> int:
    return (player + 2) % 4


def _deck_for_config(config: TichuRoundConfig) -> list[int]:
    return [
        card.card_id
        for card in DECK
        if config.special_cards or card.special is None
    ]


def _deal_to_players(
    deck: list[int],
    hands: dict[int, list[int]],
    dealer: int,
    *,
    count: int | None = None,
) -> None:
    cursor = dealer
    dealt = 0
    while deck and (count is None or dealt < count):
        hands[cursor].append(deck.pop(0))
        cursor = (cursor + 1) % 4
        dealt += 1
    if count is not None and dealt != count:
        raise AssertionError("The configured deck is too small for the initial deal")
    for hand in hands.values():
        hand.sort()


def _opening_player(hands: dict[int, list[int]], dealer: int) -> int:
    mahjong_holder = next(
        (
            player
            for player in range(4)
            if any(CARD_BY_ID[card_id].special is Special.MAHJONG for card_id in hands[player])
        ),
        None,
    )
    return dealer if mahjong_holder is None else mahjong_holder


@dataclass
class TichuRound:
    seed: int
    dealer: int
    hands: dict[int, list[int]]
    deck_remaining: list[int]
    phase: Phase = Phase.GRAND_TICHU
    current_player: int | None = 0
    config: TichuRoundConfig = field(default_factory=TichuRoundConfig)

    captured: dict[int, list[int]] = field(
        default_factory=lambda: {player: [] for player in range(4)}
    )
    discarded: list[int] = field(default_factory=list)
    pending_exchanges: dict[int, ExchangeAction] = field(default_factory=dict)
    tichu_calls: dict[int, int] = field(default_factory=dict)  # player -> 100 or 200
    has_played_card: list[bool] = field(default_factory=lambda: [False] * 4)

    wish_rank: int | None = None
    top_combination: Combination | None = None
    trick_cards: list[int] = field(default_factory=list)
    last_play_player: int | None = None
    consecutive_passes: int = 0
    finish_order: list[int] = field(default_factory=list)

    bomb_queue: list[int] = field(default_factory=list)
    bomb_resume_player: int | None = None
    bomb_close_behavior: BombCloseBehavior | None = None

    pending_dragon_winner: int | None = None
    finalize_after_dragon: bool = False

    round_scores: tuple[int, int] | None = None
    action_log: list[dict[str, Any]] = field(default_factory=list)
    public_history: list[dict[str, Any]] = field(default_factory=list)

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
        deck = _deck_for_config(config)
        rng.shuffle(deck)
        hands = {player: [] for player in range(4)}

        if config.grand_tichu:
            # The first 8 cards per player are taken one at a time. Grand Tichu
            # is decided before the ninth card is received.
            _deal_to_players(deck, hands, dealer, count=32)
        else:
            _deal_to_players(deck, hands, dealer)

        round_ = cls(
            seed=seed,
            dealer=dealer,
            hands=hands,
            deck_remaining=deck,
            config=config,
            phase=Phase.GRAND_TICHU if config.grand_tichu else Phase.PLAY,
            current_player=dealer,
        )
        if not config.grand_tichu:
            round_._start_after_deal()
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
                round_.apply(action_from_dict(raw_action["action"] if "action" in raw_action else raw_action))
            except Exception as exc:  # pragma: no cover - convenience error path
                raise ValueError(f"Replay failed at action {index}: {raw_action}") from exc
        return round_

    @property
    def terminal(self) -> bool:
        return self.phase is Phase.ROUND_OVER

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
            actions = []
            for selected in permutations(self.hands[player], 3):
                actions.append(
                    ExchangeAction(tuple(sorted(zip(recipients, selected), key=lambda x: x[0])))
                )
            return tuple(actions)

        if self.phase is Phase.DRAGON_GIFT:
            winner = self.pending_dragon_winner
            assert winner is not None
            return tuple(
                DragonGiftAction(recipient)
                for recipient in range(4)
                if team_of(recipient) != team_of(winner)
            )

        if self.phase is Phase.BOMB_WINDOW:
            if not self.config.bombs:
                return ()
            bombs = self._legal_bombs(player)
            return (BombAction(None),) + tuple(BombAction(combo) for combo in bombs)

        if self.phase is not Phase.PLAY:
            return ()

        actions: list[Action] = []
        if (
            self.config.small_tichu
            and not self.has_played_card[player]
            and player not in self.tichu_calls
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

    def apply(self, action: Action) -> None:
        if action not in self.legal_actions():
            raise ValueError(
                f"Illegal action for player {self.current_player} in {self.phase.value}: {action}"
            )
        player = self.current_player
        assert player is not None

        self.action_log.append(
            {
                "player": player,
                "phase": self.phase.value,
                "action": action_to_dict(action),
            }
        )

        if isinstance(action, GrandTichuDecision):
            self._apply_grand_tichu(player, action)
        elif isinstance(action, ExchangeAction):
            self._apply_exchange(player, action)
        elif isinstance(action, DeclareSmallTichu):
            self.tichu_calls[player] = 100
            self.public_history.append({"type": "small_tichu", "player": player})
        elif isinstance(action, PlayAction):
            self._apply_play(player, action)
        elif isinstance(action, PassAction):
            self._apply_pass(player)
        elif isinstance(action, BombAction):
            self._apply_bomb(player, action)
        elif isinstance(action, DragonGiftAction):
            self._apply_dragon_gift(player, action)
        else:  # pragma: no cover
            raise TypeError(type(action))

        self.assert_valid()

    def _apply_grand_tichu(self, player: int, action: GrandTichuDecision) -> None:
        if action.call:
            self.tichu_calls[player] = 200
            self.public_history.append({"type": "grand_tichu", "player": player})

        next_player = (player + 1) % 4
        if next_player != self.dealer:
            self.current_player = next_player
            return

        # Deal the remaining cards one at a time.
        _deal_to_players(self.deck_remaining, self.hands, self.dealer)
        self._start_after_deal()

    def _start_after_deal(self) -> None:
        if self.config.card_exchange:
            self.phase = Phase.EXCHANGE
            self.current_player = self.dealer
            return
        self._start_play()

    def _start_play(self) -> None:
        self.phase = Phase.PLAY
        self.current_player = _opening_player(self.hands, self.dealer)

    def _apply_exchange(self, player: int, action: ExchangeAction) -> None:
        self.pending_exchanges[player] = action
        self.public_history.append({"type": "exchange_submitted", "player": player})

        next_player = (player + 1) % 4
        if next_player != self.dealer:
            self.current_player = next_player
            return

        outgoing: dict[int, list[int]] = {player_id: [] for player_id in range(4)}
        incoming: dict[int, list[int]] = {player_id: [] for player_id in range(4)}
        for sender, exchange in self.pending_exchanges.items():
            for recipient, card_id in exchange.transfers:
                outgoing[sender].append(card_id)
                incoming[recipient].append(card_id)

        for sender in range(4):
            for card_id in outgoing[sender]:
                self.hands[sender].remove(card_id)
        for recipient in range(4):
            self.hands[recipient].extend(incoming[recipient])
            self.hands[recipient].sort()

        self.pending_exchanges.clear()
        self.public_history.append({"type": "exchange_complete"})
        self._start_play()

    def _apply_play(self, player: int, action: PlayAction) -> None:
        combo = action.combination
        for card_id in combo.card_ids:
            self.hands[player].remove(card_id)
        self.has_played_card[player] = True

        old_wish = self.wish_rank
        if self.config.wishes and old_wish is not None and combo.contains_normal_rank(old_wish):
            self.wish_rank = None
            self.public_history.append({"type": "wish_fulfilled", "rank": old_wish, "player": player})

        if self.config.wishes and combo.contains_special(Special.MAHJONG):
            self.wish_rank = action.wish_rank
            if action.wish_rank is not None:
                self.public_history.append(
                    {"type": "wish_made", "rank": action.wish_rank, "player": player}
                )

        self.public_history.append(
            {
                "type": "play",
                "player": player,
                "combination": combination_to_dict(combo),
                "wish_rank": action.wish_rank,
            }
        )

        if combo.is_dog:
            self.discarded.extend(combo.card_ids)
            if not self.hands[player]:
                self._record_finish(player)
                if self._finish_condition_after_play():
                    return
            target = partner_of(player)
            if not self.hands[target]:
                target = self._next_active(target)
            self._reset_trick()
            self.phase = Phase.PLAY
            self.current_player = target
            return

        self.trick_cards.extend(combo.card_ids)
        self.top_combination = combo
        self.last_play_player = player
        self.consecutive_passes = 0

        if not self.hands[player]:
            self._record_finish(player)
            if self._finish_condition_after_play():
                return

        close_behavior = (
            BombCloseBehavior.RESOLVE_TRICK
            if combo.is_bomb
            else BombCloseBehavior.CONTINUE
        )
        self._open_bomb_window(
            anchor=player,
            resume_player=self._next_active(player),
            close_behavior=close_behavior,
        )

    def _apply_pass(self, player: int) -> None:
        self.public_history.append({"type": "pass", "player": player})
        self.consecutive_passes += 1
        needed = self._passes_needed()
        close_behavior = (
            BombCloseBehavior.RESOLVE_TRICK
            if self.consecutive_passes >= needed
            else BombCloseBehavior.CONTINUE
        )
        self._open_bomb_window(
            anchor=player,
            resume_player=self._next_active(player),
            close_behavior=close_behavior,
        )

    def _apply_bomb(self, player: int, action: BombAction) -> None:
        if action.combination is None:
            self.public_history.append({"type": "decline_bomb", "player": player})
            assert self.bomb_queue and self.bomb_queue[0] == player
            self.bomb_queue.pop(0)
            if self.bomb_queue:
                self.current_player = self.bomb_queue[0]
            else:
                self._close_bomb_window()
            return

        combo = action.combination
        for card_id in combo.card_ids:
            self.hands[player].remove(card_id)
        self.has_played_card[player] = True
        self.trick_cards.extend(combo.card_ids)
        self.top_combination = combo
        self.last_play_player = player
        self.consecutive_passes = 0
        self.public_history.append(
            {
                "type": "bomb",
                "player": player,
                "combination": combination_to_dict(combo),
            }
        )

        if self.config.wishes and self.wish_rank is not None and combo.contains_normal_rank(self.wish_rank):
            old_wish = self.wish_rank
            self.wish_rank = None
            self.public_history.append({"type": "wish_fulfilled", "rank": old_wish, "player": player})

        if not self.hands[player]:
            self._record_finish(player)
            if self._finish_condition_after_play():
                return

        self._open_bomb_window(
            anchor=player,
            resume_player=self._next_active(player),
            close_behavior=BombCloseBehavior.RESOLVE_TRICK,
        )

    def _apply_dragon_gift(self, player: int, action: DragonGiftAction) -> None:
        if player != self.pending_dragon_winner:
            raise AssertionError("Only the Dragon trick winner chooses the recipient")
        self.captured[action.recipient].extend(self.trick_cards)
        self.public_history.append(
            {
                "type": "dragon_gift",
                "winner": player,
                "recipient": action.recipient,
            }
        )
        self._reset_trick()
        self.pending_dragon_winner = None

        if self.finalize_after_dragon:
            self.finalize_after_dragon = False
            self._finalize_normal_round()
            return

        self.phase = Phase.PLAY
        self.current_player = player if self.hands[player] else self._next_active(player)

    def _open_bomb_window(
        self,
        *,
        anchor: int,
        resume_player: int,
        close_behavior: BombCloseBehavior,
    ) -> None:
        if self.top_combination is None or self.top_combination.is_dog:
            self.phase = Phase.PLAY
            self.current_player = resume_player
            return

        self.bomb_resume_player = resume_player
        self.bomb_close_behavior = close_behavior
        if not self.config.bombs:
            self._close_bomb_window()
            return

        queue = self._active_cycle_after(anchor)
        if not any(self._legal_bombs(player) for player in queue):
            self._close_bomb_window()
            return

        self.phase = Phase.BOMB_WINDOW
        self.bomb_queue = queue
        self.current_player = queue[0]

    def _close_bomb_window(self) -> None:
        behavior = self.bomb_close_behavior
        resume = self.bomb_resume_player
        self.bomb_queue = []
        self.bomb_close_behavior = None
        self.bomb_resume_player = None
        if behavior is BombCloseBehavior.RESOLVE_TRICK:
            self._resolve_trick()
        else:
            self.phase = Phase.PLAY
            self.current_player = resume

    def _resolve_trick(self) -> None:
        winner = self.last_play_player
        top = self.top_combination
        if winner is None or top is None:
            raise AssertionError("Cannot resolve an empty trick")

        if top.contains_special(Special.DRAGON):
            self.phase = Phase.DRAGON_GIFT
            self.pending_dragon_winner = winner
            self.current_player = winner
            return

        self.captured[winner].extend(self.trick_cards)
        self.public_history.append({"type": "trick_won", "player": winner})
        self._reset_trick()
        self.phase = Phase.PLAY
        self.current_player = winner if self.hands[winner] else self._next_active(winner)

    def _record_finish(self, player: int) -> None:
        if player not in self.finish_order:
            self.finish_order.append(player)
            self.public_history.append(
                {"type": "went_out", "player": player, "position": len(self.finish_order)}
            )

    def _finish_condition_after_play(self) -> bool:
        if len(self.finish_order) >= 2 and team_of(self.finish_order[0]) == team_of(self.finish_order[1]):
            self._finalize_double_victory()
            return True

        if len(self.finish_order) == 3:
            if (
                self.top_combination is not None
                and self.top_combination.contains_special(Special.DRAGON)
            ):
                self.phase = Phase.DRAGON_GIFT
                self.pending_dragon_winner = self.last_play_player
                self.current_player = self.last_play_player
                self.finalize_after_dragon = True
            else:
                if self.trick_cards and self.last_play_player is not None:
                    self.captured[self.last_play_player].extend(self.trick_cards)
                    self.public_history.append(
                        {"type": "final_trick_won", "player": self.last_play_player}
                    )
                self._reset_trick()
                self._finalize_normal_round()
            return True
        return False

    def _finalize_double_victory(self) -> None:
        winning_team = team_of(self.finish_order[0])
        scores = [0, 0]
        scores[winning_team] = 200
        self._apply_tichu_scores(scores)
        self.round_scores = (scores[0], scores[1])
        self.phase = Phase.ROUND_OVER
        self.current_player = None
        self.bomb_queue = []
        self.public_history.append(
            {"type": "round_over", "double_victory": True, "scores": list(self.round_scores)}
        )

    def _finalize_normal_round(self) -> None:
        if len(self.finish_order) != 3:
            raise AssertionError("Normal round scoring requires exactly three players out")
        tailender = next(player for player in range(4) if self.hands[player])
        first_out = self.finish_order[0]

        # The tailender's remaining hand goes to the opposing team. Store the
        # physical cards with the next opponent for deterministic bookkeeping;
        # team scoring is unaffected by which opponent receives them.
        opponent_recipient = (tailender + 1) % 4
        self.captured[opponent_recipient].extend(self.hands[tailender])
        self.hands[tailender].clear()

        # The tailender's captured tricks go to the first player out.
        self.captured[first_out].extend(self.captured[tailender])
        self.captured[tailender].clear()

        scores = [0, 0]
        for player in range(4):
            scores[team_of(player)] += sum(CARD_BY_ID[card_id].points for card_id in self.captured[player])
        self._apply_tichu_scores(scores)

        self.round_scores = (scores[0], scores[1])
        self.phase = Phase.ROUND_OVER
        self.current_player = None
        self.bomb_queue = []
        self.public_history.append(
            {"type": "round_over", "double_victory": False, "scores": list(self.round_scores)}
        )

    def _apply_tichu_scores(self, scores: list[int]) -> None:
        if not self.finish_order:
            return
        winner = self.finish_order[0]
        for player, value in self.tichu_calls.items():
            scores[team_of(player)] += value if player == winner else -value

    def _combinations_in_hand(self, player: int) -> tuple[Combination, ...]:
        return generate_combinations(tuple(self.hands[player]), self.config.bombs)

    def _legal_bombs(self, player: int) -> tuple[Combination, ...]:
        if not self.config.bombs or self.top_combination is None:
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

    def _active_cycle_after(self, player: int) -> list[int]:
        return [
            candidate
            for offset in range(1, 5)
            if self.hands[candidate := (player + offset) % 4]
        ]

    def _reset_trick(self) -> None:
        self.top_combination = None
        self.trick_cards = []
        self.last_play_player = None
        self.consecutive_passes = 0

    def observation(self, player: int) -> dict[str, Any]:
        if player not in range(4):
            raise ValueError("player must be 0, 1, 2, or 3")
        return {
            "config": self.config.to_dict(),
            "phase": self.phase.value,
            "current_player": self.current_player,
            "player": player,
            "team": team_of(player),
            "hand": [
                {"card_id": card_id, "label": CARD_BY_ID[card_id].label}
                for card_id in sorted(self.hands[player])
            ],
            "hand_sizes": [len(self.hands[p]) for p in range(4)],
            "top_combination": (
                None if self.top_combination is None else combination_to_dict(self.top_combination)
            ),
            "wish_rank": self.wish_rank,
            "finish_order": list(self.finish_order),
            "tichu_calls": dict(self.tichu_calls),
            "public_history": list(self.public_history),
            "round_scores": self.round_scores,
        }

    def full_state(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "dealer": self.dealer,
            "config": self.config.to_dict(),
            "phase": self.phase.value,
            "current_player": self.current_player,
            "hands": {str(player): list(cards) for player, cards in self.hands.items()},
            "captured": {str(player): list(cards) for player, cards in self.captured.items()},
            "discarded": list(self.discarded),
            "wish_rank": self.wish_rank,
            "top_combination": (
                None if self.top_combination is None else combination_to_dict(self.top_combination)
            ),
            "trick_cards": list(self.trick_cards),
            "finish_order": list(self.finish_order),
            "tichu_calls": {str(player): value for player, value in self.tichu_calls.items()},
            "round_scores": None if self.round_scores is None else list(self.round_scores),
        }

    def save_replay(self, path: str) -> None:
        payload = {
            "seed": self.seed,
            "dealer": self.dealer,
            "config": self.config.to_dict(),
            "actions": self.action_log,
            "final_state": self.full_state(),
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)

    def assert_valid(self) -> None:
        zones: list[int] = []
        for hand in self.hands.values():
            if len(hand) != len(set(hand)):
                raise AssertionError("A hand contains a duplicate physical card")
            zones.extend(hand)
        for pile in self.captured.values():
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
        if self.phase is Phase.BOMB_WINDOW and not self.config.bombs:
            raise AssertionError("Bomb window phase is disabled")
        if self.phase is Phase.DRAGON_GIFT and not self.config.special_cards:
            raise AssertionError("Dragon gift phase requires special cards")
        if self.wish_rank is not None and self.wish_rank not in range(2, 15):
            raise AssertionError("Invalid Mahjong wish")
        if self.wish_rank is not None and not self.config.wishes:
            raise AssertionError("Mahjong wishes are disabled")
        for player, value in self.tichu_calls.items():
            if player not in range(4):
                raise AssertionError("Invalid Tichu caller")
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
