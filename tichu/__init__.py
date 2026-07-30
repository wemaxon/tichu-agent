from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "BombAction": ".actions",
    "DeclareSmallTichu": ".actions",
    "DragonGiftAction": ".actions",
    "ExchangeAction": ".actions",
    "GrandTichuDecision": ".actions",
    "PassAction": ".actions",
    "PlayAction": ".actions",
    "CARD_BY_ID": ".cards",
    "CARD_CLASS_NAMES": ".cards",
    "Card": ".cards",
    "DECK": ".cards",
    "Rank": ".cards",
    "Special": ".cards",
    "Suit": ".cards",
    "Combination": ".combinations",
    "CombinationKind": ".combinations",
    "generate_bombs": ".combinations",
    "generate_combinations": ".combinations",
    "generate_combinations_for_context": ".combinations",
    "TichuRoundConfig": ".config",
    "RoundObservation": ".observations",
    "Phase": ".round",
    "TichuRound": ".round",
    "partner_of": ".round",
    "team_of": ".round",
    "TichuMatch": ".match",
}

__all__ = tuple(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
