from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from ml import CandidateBatch, batch_candidates, batch_env_states


@dataclass(frozen=True)
class _Candidate:
    features: tuple[float, ...]


@dataclass(frozen=True)
class _State:
    terminal: bool
    observation_features: tuple[float, ...] | None
    encoded_actions: tuple[_Candidate, ...]


@pytest.mark.parametrize("candidate_count", [2, 2184])
def test_batch_candidates_supports_variable_action_counts(
    candidate_count: int,
) -> None:
    batch = batch_candidates(
        observations=((1.0, 2.0, 3.0),),
        action_groups=(
            tuple((float(index), 1.0) for index in range(candidate_count)),
        ),
    )

    assert batch.observations.shape == (1, 3)
    assert batch.actions.shape == (candidate_count, 2)
    assert batch.candidate_owner.tolist() == [0] * candidate_count
    assert batch.offsets.tolist() == [0, candidate_count]
    assert batch.candidate_counts.tolist() == [candidate_count]


def test_batch_candidates_preserves_contiguous_state_groups() -> None:
    batch = batch_candidates(
        observations=((1.0, 0.0), (0.0, 1.0)),
        action_groups=(
            ((1.0,), (2.0,)),
            ((3.0,), (4.0,), (5.0,)),
        ),
        dtype=torch.float64,
    )

    assert batch.dtype == torch.float64
    assert batch.state_count == 2
    assert batch.candidate_count == 5
    assert batch.actions[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert batch.candidate_owner.tolist() == [0, 0, 1, 1, 1]
    assert batch.offsets.tolist() == [0, 2, 5]


def test_batch_env_states_uses_only_policy_facing_features() -> None:
    states = (
        _State(
            terminal=False,
            observation_features=(1.0, 2.0),
            encoded_actions=(_Candidate((3.0,)), _Candidate((4.0,))),
        ),
        _State(
            terminal=False,
            observation_features=(5.0, 6.0),
            encoded_actions=(_Candidate((7.0,)),),
        ),
    )

    batch = batch_env_states(states)

    assert batch.observations.tolist() == [[1.0, 2.0], [5.0, 6.0]]
    assert batch.actions.tolist() == [[3.0], [4.0], [7.0]]
    assert batch.offsets.tolist() == [0, 2, 3]


def test_batch_env_states_rejects_terminal_and_empty_action_states() -> None:
    terminal = _State(True, None, ())
    empty = _State(False, (1.0,), ())

    with pytest.raises(ValueError, match="terminal"):
        batch_env_states((terminal,))
    with pytest.raises(ValueError, match="no candidates"):
        batch_env_states((empty,))


def test_candidate_batch_rejects_inconsistent_owner_mapping() -> None:
    with pytest.raises(ValueError, match="does not match offsets"):
        CandidateBatch(
            observations=torch.zeros((2, 3)),
            actions=torch.zeros((3, 2)),
            candidate_owner=torch.tensor([0, 1, 1]),
            offsets=torch.tensor([0, 2, 3]),
        )
