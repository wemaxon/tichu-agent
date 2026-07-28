from __future__ import annotations

from typing import Protocol

from tichu.actions import Action
from tichu.round import TichuRound


class Agent(Protocol):
    def choose_action(self, round_: TichuRound) -> Action:
        ...
