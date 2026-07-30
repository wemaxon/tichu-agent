from __future__ import annotations

import pytest
import torch

from ml import RaggedCategorical


def test_ragged_probabilities_normalize_each_group() -> None:
    distribution = RaggedCategorical(
        logits=torch.tensor([0.0, 1.0, -1.0, 0.0, 2.0]),
        offsets=torch.tensor([0, 2, 5]),
    )

    probabilities = distribution.probabilities

    assert probabilities[:2].sum().item() == pytest.approx(1.0)
    assert probabilities[2:].sum().item() == pytest.approx(1.0)
    assert distribution.entropy().shape == (2,)


def test_ragged_mode_and_log_prob_use_state_local_indices() -> None:
    logits = torch.tensor([0.0, 1.0, -1.0, 0.0, 2.0])
    distribution = RaggedCategorical(
        logits=logits,
        offsets=torch.tensor([0, 2, 5]),
    )

    modes = distribution.mode()
    selected_log_probs = distribution.log_prob(torch.tensor([1, 2]))

    assert modes.tolist() == [1, 2]
    assert selected_log_probs[0] == torch.log_softmax(logits[:2], dim=0)[1]
    assert selected_log_probs[1] == torch.log_softmax(logits[2:], dim=0)[2]


def test_ragged_sampling_never_leaves_a_local_action_group() -> None:
    distribution = RaggedCategorical(
        logits=torch.zeros(7),
        offsets=torch.tensor([0, 2, 7]),
    )
    generator = torch.Generator().manual_seed(123)

    for _ in range(100):
        sample = distribution.sample(generator=generator)
        assert 0 <= sample[0].item() < 2
        assert 0 <= sample[1].item() < 5


def test_ragged_distribution_rejects_empty_groups_and_invalid_indices() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        RaggedCategorical(
            logits=torch.zeros(2),
            offsets=torch.tensor([0, 0, 2]),
        )

    distribution = RaggedCategorical(
        logits=torch.zeros(2),
        offsets=torch.tensor([0, 2]),
    )
    with pytest.raises(IndexError, match="out of range"):
        distribution.log_prob(torch.tensor([2]))
