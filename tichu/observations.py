from __future__ import annotations

from typing import Any, Literal, TypedDict


PhaseName = Literal[
    "grand_tichu",
    "exchange",
    "play",
    "bomb_window",
    "dragon_gift",
    "round_over",
]


class CardObservation(TypedDict):
    card_id: int
    label: str


class CombinationObservation(TypedDict):
    card_ids: list[int]
    kind: str
    primary_rank: float
    length: int
    phoenix_as: int | None


class RoundObservation(TypedDict):
    config: dict[str, bool]
    phase: PhaseName
    current_player: int | None
    player: int
    team: int
    hand: list[CardObservation]
    hand_sizes: list[int]
    top_combination: CombinationObservation | None
    wish_rank: int | None
    finish_order: list[int]
    tichu_calls: dict[int, int]
    public_history: list[dict[str, Any]]
    round_scores: tuple[int, int] | None
