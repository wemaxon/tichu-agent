from __future__ import annotations

import random
from collections.abc import Sequence

from tichu.actions import (
    Action,
    BombAction,
    DeclareSmallTichu,
    ExchangeAction,
    GrandTichuDecision,
    PlayAction,
)
from tichu.observations import RoundObservation


class RandomAgent:
    """A baseline"""

    def __init__(
        self,
        seed: int = 0,
        *,
        call_grand_probability: float = 0.0,
        call_small_probability: float = 0.0,
        bomb_probability: float = 0.2,
    ) -> None:
        self.rng = random.Random(seed)
        self.call_grand_probability = call_grand_probability
        self.call_small_probability = call_small_probability
        self.bomb_probability = bomb_probability

    def choose_action(
        self,
        observation: RoundObservation,
        legal_actions: Sequence[Action],
    ) -> Action:
        player = observation["current_player"]
        if player is None:
            raise RuntimeError("No action is available")
        actions = list(legal_actions)
        if not actions:
            raise RuntimeError("The simulator produced no legal actions")

        phase = observation["phase"]

        if phase == "grand_tichu":
            return GrandTichuDecision(self.rng.random() < self.call_grand_probability)

        if phase == "exchange":
            recipients = [other for other in range(4) if other != player]
            hand = [card["card_id"] for card in observation["hand"]]
            selected = self.rng.sample(hand, 3)
            return ExchangeAction(tuple(sorted(zip(recipients, selected))))

        if phase == "bomb_window":
            bombs = [action for action in actions if isinstance(action, BombAction) and action.combination]
            if bombs and self.rng.random() < self.bomb_probability:
                return self.rng.choice(bombs)
            return BombAction(None)

        if phase == "play":
            declarations = [action for action in actions if isinstance(action, DeclareSmallTichu)]
            if declarations and self.rng.random() < self.call_small_probability:
                return declarations[0]
            non_declarations = [action for action in actions if not isinstance(action, DeclareSmallTichu)]
            # Avoid overweighting Mahjong plays merely because each has 14 wish variants.
            groups: dict[tuple, list] = {}
            for action in non_declarations:
                if isinstance(action, PlayAction):
                    key = ("play", action.combination)
                else:
                    key = (type(action).__name__,)
                groups.setdefault(key, []).append(action)
            group = self.rng.choice(list(groups.values()))
            return self.rng.choice(group)

        return self.rng.choice(actions)
