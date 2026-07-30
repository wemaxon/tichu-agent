from __future__ import annotations

import torch
from torch import Tensor


class RaggedCategorical:
    """Independent categorical distributions stored as contiguous logit groups."""

    def __init__(self, logits: Tensor, offsets: Tensor) -> None:
        if logits.ndim != 1:
            raise ValueError("logits must have shape [candidates]")
        if not logits.is_floating_point():
            raise TypeError("logits must use a floating-point dtype")
        if offsets.ndim != 1:
            raise ValueError("offsets must have shape [groups + 1]")
        if offsets.dtype != torch.long:
            raise TypeError("offsets must use torch.long")
        if logits.device != offsets.device:
            raise ValueError("logits and offsets must use the same device")
        if offsets.shape[0] < 2:
            raise ValueError("at least one categorical group is required")
        if offsets[0].item() != 0 or offsets[-1].item() != logits.shape[0]:
            raise ValueError("offsets must span all logits")

        counts = offsets[1:] - offsets[:-1]
        if torch.any(counts <= 0).item():
            raise ValueError("categorical groups cannot be empty")

        self.logits = logits
        self.offsets = offsets
        self._candidate_owner = torch.repeat_interleave(
            torch.arange(counts.shape[0], device=logits.device),
            counts,
        )

    @property
    def group_count(self) -> int:
        return self.offsets.shape[0] - 1

    @property
    def log_probabilities(self) -> Tensor:
        group_maxima = self._group_max(self.logits)
        shifted_logits = self.logits - group_maxima.index_select(
            0,
            self._candidate_owner,
        )
        normalizers = torch.zeros(
            self.group_count,
            dtype=self.logits.dtype,
            device=self.logits.device,
        ).scatter_add(
            0,
            self._candidate_owner,
            shifted_logits.exp(),
        )
        return shifted_logits - normalizers.log().index_select(
            0,
            self._candidate_owner,
        )

    @property
    def probabilities(self) -> Tensor:
        return self.log_probabilities.exp()

    def entropy(self) -> Tensor:
        log_probabilities = self.log_probabilities
        return torch.zeros(
            self.group_count,
            dtype=self.logits.dtype,
            device=self.logits.device,
        ).scatter_add(
            0,
            self._candidate_owner,
            -(log_probabilities.exp() * log_probabilities),
        )

    def mode(self) -> Tensor:
        return self._group_argmax(self.logits)

    def sample(self, *, generator: torch.Generator | None = None) -> Tensor:
        exponential_noise = torch.empty_like(self.logits).exponential_(
            generator=generator
        )
        gumbel_noise = -exponential_noise.clamp_min(
            torch.finfo(self.logits.dtype).tiny
        ).log()
        return self._group_argmax(self.logits + gumbel_noise)

    def log_prob(self, local_indices: Tensor) -> Tensor:
        if local_indices.ndim != 1 or local_indices.shape[0] != self.group_count:
            raise ValueError("local_indices must contain one index per group")
        if local_indices.dtype != torch.long:
            raise TypeError("local_indices must use torch.long")
        if local_indices.device != self.logits.device:
            raise ValueError("local_indices and logits must use the same device")

        counts = self.offsets[1:] - self.offsets[:-1]
        if torch.any(local_indices < 0).item() or torch.any(
            local_indices >= counts
        ).item():
            raise IndexError("local categorical index out of range")

        global_indices = self.offsets[:-1] + local_indices
        return self.log_probabilities.index_select(0, global_indices)

    def _group_max(self, values: Tensor) -> Tensor:
        return torch.full(
            (self.group_count,),
            -torch.inf,
            dtype=values.dtype,
            device=values.device,
        ).scatter_reduce(
            0,
            self._candidate_owner,
            values,
            reduce="amax",
            include_self=True,
        )

    def _group_argmax(self, values: Tensor) -> Tensor:
        maxima = self._group_max(values).index_select(0, self._candidate_owner)
        global_indices = torch.arange(
            values.shape[0],
            dtype=torch.long,
            device=values.device,
        )
        sentinel = values.shape[0]
        winners = torch.where(
            values == maxima,
            global_indices,
            sentinel,
        )
        selected_global = torch.full(
            (self.group_count,),
            sentinel,
            dtype=torch.long,
            device=values.device,
        ).scatter_reduce(
            0,
            self._candidate_owner,
            winners,
            reduce="amin",
            include_self=True,
        )
        return selected_global - self.offsets[:-1]
