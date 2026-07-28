from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TichuRoundConfig:
    grand_tichu: bool = True
    small_tichu: bool = True
    card_exchange: bool = True
    bombs: bool = True
    wishes: bool = True
    special_cards: bool = True

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TichuRoundConfig":
        if data is None:
            return cls()
        known = {field.name for field in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown round config option(s): {sorted(unknown)}")
        return cls(**dict(data))
