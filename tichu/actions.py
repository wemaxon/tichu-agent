from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypeAlias

from .combinations import Combination, CombinationKind


@dataclass(frozen=True)
class GrandTichuDecision:
    call: bool


@dataclass(frozen=True)
class ExchangeAction:
    """One face-down card sent to each of the other three players."""

    transfers: tuple[tuple[int, int], ...]  # (recipient, card_id), sorted by recipient


@dataclass(frozen=True)
class DeclareSmallTichu:
    pass


@dataclass(frozen=True)
class PlayAction:
    combination: Combination
    wish_rank: int | None = None


@dataclass(frozen=True)
class PassAction:
    pass


@dataclass(frozen=True)
class BombAction:
    combination: Combination | None  # None means decline the bomb opportunity


@dataclass(frozen=True)
class DragonGiftAction:
    recipient: int


Action: TypeAlias = (
    GrandTichuDecision
    | ExchangeAction
    | DeclareSmallTichu
    | PlayAction
    | PassAction
    | BombAction
    | DragonGiftAction
)


def combination_to_dict(combo: Combination) -> dict[str, Any]:
    return {
        "card_ids": list(combo.card_ids),
        "kind": combo.kind.value,
        "primary_rank": combo.primary_rank,
        "length": combo.length,
        "phoenix_as": combo.phoenix_as,
    }


def combination_from_dict(data: dict[str, Any]) -> Combination:
    return Combination(
        card_ids=tuple(int(value) for value in data["card_ids"]),
        kind=CombinationKind(data["kind"]),
        primary_rank=float(data["primary_rank"]),
        length=int(data["length"]),
        phoenix_as=(None if data.get("phoenix_as") is None else int(data["phoenix_as"])),
    )


def action_to_dict(action: Action) -> dict[str, Any]:
    if isinstance(action, GrandTichuDecision):
        return {"type": "grand_tichu", "call": action.call}
    if isinstance(action, ExchangeAction):
        return {"type": "exchange", "transfers": [list(x) for x in action.transfers]}
    if isinstance(action, DeclareSmallTichu):
        return {"type": "small_tichu"}
    if isinstance(action, PlayAction):
        return {
            "type": "play",
            "combination": combination_to_dict(action.combination),
            "wish_rank": action.wish_rank,
        }
    if isinstance(action, PassAction):
        return {"type": "pass"}
    if isinstance(action, BombAction):
        return {
            "type": "bomb",
            "combination": (
                None if action.combination is None else combination_to_dict(action.combination)
            ),
        }
    if isinstance(action, DragonGiftAction):
        return {"type": "dragon_gift", "recipient": action.recipient}
    raise TypeError(f"Unsupported action: {type(action)!r}")


def action_from_dict(data: dict[str, Any]) -> Action:
    action_type = data["type"]
    if action_type == "grand_tichu":
        return GrandTichuDecision(bool(data["call"]))
    if action_type == "exchange":
        return ExchangeAction(
            tuple(sorted((int(recipient), int(card_id)) for recipient, card_id in data["transfers"]))
        )
    if action_type == "small_tichu":
        return DeclareSmallTichu()
    if action_type == "play":
        return PlayAction(
            combination_from_dict(data["combination"]),
            None if data.get("wish_rank") is None else int(data["wish_rank"]),
        )
    if action_type == "pass":
        return PassAction()
    if action_type == "bomb":
        raw_combo = data.get("combination")
        return BombAction(None if raw_combo is None else combination_from_dict(raw_combo))
    if action_type == "dragon_gift":
        return DragonGiftAction(int(data["recipient"]))
    raise ValueError(f"Unknown action type: {action_type}")
