from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias

from tichu.actions import Action
from tichu.observations import RoundObservation


LegalActions: TypeAlias = Sequence[Action]


class Agent(Protocol):
    def choose_action(self, observation: RoundObservation, legal_actions: LegalActions) -> Action:
        ...
