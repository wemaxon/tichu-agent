from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EncodedCandidate(Protocol):
    @property
    def features(self) -> Sequence[float]:
        ...


class EncodedPolicyState(Protocol):
    @property
    def terminal(self) -> bool:
        ...

    @property
    def observation_features(self) -> Sequence[float] | None:
        ...

    @property
    def encoded_actions(self) -> Sequence[EncodedCandidate]:
        ...


class PolicyDecisionLike(Protocol):
    @property
    def action_index(self) -> int:
        ...


class InferencePolicy(Protocol):
    def decide(
        self,
        state: EncodedPolicyState,
        *,
        deterministic: bool = False,
    ) -> PolicyDecisionLike:
        ...
