from __future__ import annotations

from dataclasses import dataclass
import math

import pytest
import torch

from ml import (
    ACTION_FEATURE_SIZE,
    OBSERVATION_FEATURE_SIZE,
    ActorCriticConfig,
    ParametricActorCritic,
    ParametricPolicy,
    TichuEnv,
    batch_env_states,
)
from tichu.config import TichuRoundConfig


@dataclass(frozen=True)
class _Candidate:
    features: tuple[float, ...]


@dataclass(frozen=True)
class _State:
    terminal: bool
    observation_features: tuple[float, ...] | None
    encoded_actions: tuple[_Candidate, ...]


def _model() -> ParametricActorCritic:
    return ParametricActorCritic(
        ActorCriticConfig(
            observation_size=OBSERVATION_FEATURE_SIZE,
            action_size=ACTION_FEATURE_SIZE,
            state_hidden_size=16,
            action_hidden_size=12,
            embedding_size=8,
            scorer_hidden_size=6,
            value_hidden_size=4,
        )
    )


@pytest.mark.parametrize("out_of_turn_bombs", [False, True])
def test_policy_selects_a_legal_action_in_both_bomb_modes(
    out_of_turn_bombs: bool,
) -> None:
    state = TichuEnv(
        seed=4,
        config=TichuRoundConfig(out_of_turn_bombs=out_of_turn_bombs),
    ).state()
    policy = ParametricPolicy(_model())

    decision = policy.decide(state, deterministic=False)

    assert 0 <= decision.action_index < state.action_count
    assert math.isfinite(decision.log_probability)
    assert math.isfinite(decision.value)
    assert math.isfinite(decision.entropy)


def test_deterministic_policy_selects_the_highest_scoring_local_action() -> None:
    torch.manual_seed(3)
    state = TichuEnv(seed=5).state()
    model = _model()
    batch = batch_env_states((state,))
    expected_index = int(model(batch).logits.argmax().item())

    decision = ParametricPolicy(model).decide(state, deterministic=True)

    assert decision.action_index == expected_index


def test_policy_depends_only_on_encoded_state_contract() -> None:
    state = _State(
        terminal=False,
        observation_features=(0.0,) * OBSERVATION_FEATURE_SIZE,
        encoded_actions=(
            _Candidate((0.0,) * ACTION_FEATURE_SIZE),
            _Candidate((1.0,) * ACTION_FEATURE_SIZE),
        ),
    )

    decision = ParametricPolicy(_model()).decide(state, deterministic=True)

    assert decision.action_index in (0, 1)


def test_policy_rejects_terminal_states() -> None:
    terminal = _State(
        terminal=True,
        observation_features=None,
        encoded_actions=(),
    )

    with pytest.raises(ValueError, match="terminal"):
        ParametricPolicy(_model()).decide(terminal)
