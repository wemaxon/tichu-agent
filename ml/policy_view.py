from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

from tichu.observations import BombCloseBehaviorName, PhaseName, RoundObservation

PLAYER_COUNT = 4
POLICY_CONFIG_KEYS: tuple[str, ...] = (
    "grand_tichu",
    "small_tichu",
    "card_exchange",
    "bombs",
    "wishes",
    "special_cards",
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PolicyCombination:
    card_ids: tuple[int, ...]
    kind: str
    primary_rank: float
    length: int
    phoenix_as: int | None


@dataclass(frozen=True, slots=True)
class PolicyExchangeCard:
    """An exchange card with an actor-relative player identifier."""

    player: int
    card_id: int


@dataclass(frozen=True, slots=True)
class PolicyExchange:
    sent: tuple[PolicyExchangeCard, ...]
    received: tuple[PolicyExchangeCard, ...]


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    """Player-safe game information with the observer canonicalized to seat zero.

    Relative seats follow play order: 0 is the observer, 1 the next seat,
    2 the observer's partner, and 3 the previous seat.
    """

    config: tuple[bool, ...]
    phase: PhaseName
    current_player: int | None
    dealer: int
    hand_card_ids: tuple[int, ...]
    hand_sizes: tuple[int, int, int, int]
    played_card_ids: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]
    captured_card_ids: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]
    captured_points: tuple[int, int, int, int]
    current_trick_card_ids: tuple[int, ...]
    top_combination: PolicyCombination | None
    top_player: int | None
    consecutive_passes: int
    bomb_queue: tuple[int, ...]
    bomb_resume_player: int | None
    bomb_close_behavior: BombCloseBehaviorName | None
    wish_rank: int | None
    finish_order: tuple[int, ...]
    tichu_calls: tuple[int | None, int | None, int | None, int | None]
    has_played_card: tuple[bool, bool, bool, bool]
    private_exchange: PolicyExchange | None


def relative_player(player: int, observer: int) -> int:
    _validate_player(player, "player")
    _validate_player(observer, "observer")
    return (player - observer) % PLAYER_COUNT


def absolute_player(player: int, observer: int) -> int:
    _validate_player(player, "relative player")
    _validate_player(observer, "observer")
    return (player + observer) % PLAYER_COUNT


def rotate_player_values(values: Sequence[T], observer: int) -> tuple[T, T, T, T]:
    _validate_player(observer, "observer")
    if len(values) != PLAYER_COUNT:
        raise ValueError("Player values must contain exactly four entries")
    rotated = tuple(values[absolute_player(player, observer)] for player in range(PLAYER_COUNT))
    return rotated[0], rotated[1], rotated[2], rotated[3]


def policy_view(observation: RoundObservation) -> PolicyObservation:
    """Convert an absolute-seat round observation into a canonical policy view."""

    observer = observation["player"]
    _validate_player(observer, "observation player")

    hand_sizes = rotate_player_values(observation["hand_sizes"], observer)
    played_card_ids = rotate_player_values(
        tuple(tuple(cards) for cards in observation["played_card_ids"]),
        observer,
    )
    captured_card_ids = rotate_player_values(
        tuple(tuple(cards) for cards in observation["captured_card_ids"]),
        observer,
    )
    captured_points = rotate_player_values(observation["captured_points"], observer)
    has_played_card = rotate_player_values(observation["has_played_card"], observer)
    tichu_calls = rotate_player_values(
        tuple(observation["tichu_calls"].get(player) for player in range(PLAYER_COUNT)),
        observer,
    )

    top = observation["top_combination"]
    top_combination = (
        None
        if top is None
        else PolicyCombination(
            card_ids=tuple(top["card_ids"]),
            kind=top["kind"],
            primary_rank=float(top["primary_rank"]),
            length=int(top["length"]),
            phoenix_as=top["phoenix_as"],
        )
    )

    private_exchange = observation["private_exchange"]
    relative_exchange = (
        None
        if private_exchange is None
        else PolicyExchange(
            sent=tuple(
                PolicyExchangeCard(
                    player=relative_player(transfer["player"], observer),
                    card_id=transfer["card_id"],
                )
                for transfer in private_exchange["sent"]
            ),
            received=tuple(
                PolicyExchangeCard(
                    player=relative_player(transfer["player"], observer),
                    card_id=transfer["card_id"],
                )
                for transfer in private_exchange["received"]
            ),
        )
    )

    current_player = observation["current_player"]
    top_player = observation["top_player"]
    bomb_resume_player = observation["bomb_resume_player"]
    return PolicyObservation(
        config=tuple(bool(observation["config"].get(key, False)) for key in POLICY_CONFIG_KEYS),
        phase=observation["phase"],
        current_player=(
            None if current_player is None else relative_player(current_player, observer)
        ),
        dealer=relative_player(observation["dealer"], observer),
        hand_card_ids=tuple(card["card_id"] for card in observation["hand"]),
        hand_sizes=hand_sizes,
        played_card_ids=played_card_ids,
        captured_card_ids=captured_card_ids,
        captured_points=captured_points,
        current_trick_card_ids=tuple(observation["current_trick_card_ids"]),
        top_combination=top_combination,
        top_player=None if top_player is None else relative_player(top_player, observer),
        consecutive_passes=int(observation["consecutive_passes"]),
        bomb_queue=tuple(
            relative_player(player, observer) for player in observation["bomb_queue"]
        ),
        bomb_resume_player=(
            None
            if bomb_resume_player is None
            else relative_player(bomb_resume_player, observer)
        ),
        bomb_close_behavior=observation["bomb_close_behavior"],
        wish_rank=observation["wish_rank"],
        finish_order=tuple(
            relative_player(player, observer) for player in observation["finish_order"]
        ),
        tichu_calls=tichu_calls,
        has_played_card=has_played_card,
        private_exchange=relative_exchange,
    )


def _validate_player(player: int, context: str) -> None:
    if player not in range(PLAYER_COUNT):
        raise ValueError(f"{context} must be between 0 and {PLAYER_COUNT - 1}: {player}")
