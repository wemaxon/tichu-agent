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
BombCloseBehaviorName = Literal["continue", "resolve_trick"]


class CardObservation(TypedDict):
    card_id: int
    label: str


class CombinationObservation(TypedDict):
    card_ids: list[int]
    kind: str
    primary_rank: float
    length: int
    phoenix_as: int | None


class ExchangeCardObservation(TypedDict):
    """One private exchange card and the player on its other end."""

    player: int
    card_id: int


class PrivateExchangeObservation(TypedDict):
    """The observing player's own sent and received exchange cards."""

    sent: list[ExchangeCardObservation]
    received: list[ExchangeCardObservation]


class RoundObservation(TypedDict):
    config: dict[str, bool]
    phase: PhaseName
    current_player: int | None
    player: int
    team: int
    dealer: int
    hand: list[CardObservation]
    hand_sizes: list[int]
    played_card_ids: list[list[int]]
    captured_card_ids: list[list[int]]
    captured_points: list[int]
    current_trick_card_ids: list[int]
    top_combination: CombinationObservation | None
    top_player: int | None
    consecutive_passes: int
    bomb_queue: list[int]
    bomb_resume_player: int | None
    bomb_close_behavior: BombCloseBehaviorName | None
    wish_rank: int | None
    finish_order: list[int]
    tichu_calls: dict[int, int]
    has_played_card: list[bool]
    private_exchange: PrivateExchangeObservation | None
    public_history: list[dict[str, Any]]
    round_scores: tuple[int, int] | None
