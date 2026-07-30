from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class EncodedCandidate(Protocol):
    features: Sequence[float]


class EncodedPolicyState(Protocol):
    terminal: bool
    observation_features: Sequence[float] | None
    encoded_actions: Sequence[EncodedCandidate]


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    """A batch of states and their contiguous, variable-sized action groups."""

    observations: Tensor
    actions: Tensor
    candidate_owner: Tensor
    offsets: Tensor

    def __post_init__(self) -> None:
        if self.observations.ndim != 2:
            raise ValueError("observations must have shape [states, features]")
        if self.actions.ndim != 2:
            raise ValueError("actions must have shape [candidates, features]")
        if self.candidate_owner.ndim != 1:
            raise ValueError("candidate_owner must have shape [candidates]")
        if self.offsets.ndim != 1:
            raise ValueError("offsets must have shape [states + 1]")
        if self.observations.shape[0] == 0:
            raise ValueError("a candidate batch must contain at least one state")
        if self.observations.shape[1] == 0:
            raise ValueError("observation feature size must be positive")
        if self.actions.shape[1] == 0:
            raise ValueError("action feature size must be positive")
        if not self.observations.is_floating_point():
            raise TypeError("observations must use a floating-point dtype")
        if not self.actions.is_floating_point():
            raise TypeError("actions must use a floating-point dtype")
        if self.observations.dtype != self.actions.dtype:
            raise TypeError("observations and actions must use the same dtype")
        if self.candidate_owner.dtype != torch.long:
            raise TypeError("candidate_owner must use torch.long")
        if self.offsets.dtype != torch.long:
            raise TypeError("offsets must use torch.long")

        devices = {
            self.observations.device,
            self.actions.device,
            self.candidate_owner.device,
            self.offsets.device,
        }
        if len(devices) != 1:
            raise ValueError("all candidate batch tensors must use the same device")

        state_count = self.observations.shape[0]
        candidate_count = self.actions.shape[0]
        if self.candidate_owner.shape[0] != candidate_count:
            raise ValueError("candidate_owner must contain one entry per candidate")
        if self.offsets.shape[0] != state_count + 1:
            raise ValueError("offsets must contain one boundary per state plus the end")
        if self.offsets[0].item() != 0:
            raise ValueError("offsets must start at zero")
        if self.offsets[-1].item() != candidate_count:
            raise ValueError("offsets must end at the candidate count")

        counts = self.offsets[1:] - self.offsets[:-1]
        if torch.any(counts <= 0).item():
            raise ValueError("every state must have at least one candidate")

        expected_owner = torch.repeat_interleave(
            torch.arange(state_count, device=self.device),
            counts,
        )
        if not torch.equal(self.candidate_owner, expected_owner):
            raise ValueError("candidate_owner does not match offsets")

    @property
    def device(self) -> torch.device:
        return self.observations.device

    @property
    def dtype(self) -> torch.dtype:
        return self.observations.dtype

    @property
    def state_count(self) -> int:
        return self.observations.shape[0]

    @property
    def candidate_count(self) -> int:
        return self.actions.shape[0]

    @property
    def candidate_counts(self) -> Tensor:
        return self.offsets[1:] - self.offsets[:-1]

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> CandidateBatch:
        target_dtype = self.dtype if dtype is None else dtype
        return CandidateBatch(
            observations=self.observations.to(device=device, dtype=target_dtype),
            actions=self.actions.to(device=device, dtype=target_dtype),
            candidate_owner=self.candidate_owner.to(device=device),
            offsets=self.offsets.to(device=device),
        )


def batch_candidates(
    observations: Sequence[Sequence[float]],
    action_groups: Sequence[Sequence[Sequence[float]]],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> CandidateBatch:
    """Pack feature sequences without padding or imposing a maximum action count."""

    if not observations:
        raise ValueError("at least one observation is required")
    if len(observations) != len(action_groups):
        raise ValueError("observations and action_groups must contain the same states")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")

    observation_size = len(observations[0])
    if observation_size == 0:
        raise ValueError("observation feature size must be positive")

    action_size: int | None = None
    flat_actions: list[Sequence[float]] = []
    counts: list[int] = []
    for state_index, (observation, candidates) in enumerate(
        zip(observations, action_groups, strict=True)
    ):
        if len(observation) != observation_size:
            raise ValueError(
                f"observation {state_index} has {len(observation)} features; "
                f"expected {observation_size}"
            )
        if not candidates:
            raise ValueError(f"state {state_index} has no candidates")

        counts.append(len(candidates))
        for candidate_index, candidate in enumerate(candidates):
            if action_size is None:
                action_size = len(candidate)
                if action_size == 0:
                    raise ValueError("action feature size must be positive")
            if len(candidate) != action_size:
                raise ValueError(
                    f"candidate {candidate_index} for state {state_index} has "
                    f"{len(candidate)} features; expected {action_size}"
                )
            flat_actions.append(candidate)

    if action_size is None:
        raise RuntimeError("non-empty action groups produced no action features")

    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)

    return CandidateBatch(
        observations=torch.tensor(observations, dtype=dtype, device=device),
        actions=torch.tensor(flat_actions, dtype=dtype, device=device),
        candidate_owner=torch.repeat_interleave(
            torch.arange(len(observations), dtype=torch.long, device=device),
            torch.tensor(counts, dtype=torch.long, device=device),
        ),
        offsets=torch.tensor(offsets, dtype=torch.long, device=device),
    )


def batch_env_states(
    states: Sequence[EncodedPolicyState],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> CandidateBatch:
    """Adapt policy-facing environment states to the tensor-only batch format."""

    observations: list[Sequence[float]] = []
    action_groups: list[list[Sequence[float]]] = []
    for state_index, state in enumerate(states):
        if state.terminal:
            raise ValueError(f"state {state_index} is terminal")
        if state.observation_features is None:
            raise ValueError(f"state {state_index} has no observation features")
        observations.append(state.observation_features)
        action_groups.append(
            [candidate.features for candidate in state.encoded_actions]
        )

    return batch_candidates(
        observations,
        action_groups,
        device=device,
        dtype=dtype,
    )
