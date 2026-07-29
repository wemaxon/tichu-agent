from __future__ import annotations

import pytest

from ml import (
    ACTION_FEATURE_SIZE,
    OBSERVATION_FEATURE_SIZE,
    TichuEnv,
    encode_env_state,
)
from tichu.actions import PassAction
from tichu.config import TichuRoundConfig
from tichu.round import Phase

from .helpers import make_round


def _fast_config() -> TichuRoundConfig:
    return TichuRoundConfig(
        grand_tichu=False,
        small_tichu=False,
        card_exchange=False,
        bombs=True,
        wishes=False,
        special_cards=False,
    )


def test_env_reset_returns_encoded_turn_state() -> None:
    env = TichuEnv(config=_fast_config(), seed=0, dealer=0)
    state = env.reset(seed=1, dealer=2)

    assert env.seed == 1
    assert env.dealer == 2
    assert state.round is env.round
    assert state.steps == 0
    assert state.player == state.round.current_player
    assert state.observation is not None
    assert state.observation_features is not None
    assert len(state.observation_features) == OBSERVATION_FEATURE_SIZE
    assert state.action_count == len(state.legal_actions)
    assert len(state.encoded_actions) == len(state.legal_actions)
    assert all(len(action.features) == ACTION_FEATURE_SIZE for action in state.encoded_actions)
    assert state.rewards == (0.0, 0.0, 0.0, 0.0)
    assert not state.terminal


def test_env_steps_by_legal_action_index() -> None:
    env = TichuEnv(config=_fast_config(), seed=0, dealer=0)
    before = env.state()

    after = env.step(0)

    assert after.steps == 1
    assert env.steps == 1
    assert after.round is env.round
    assert after.round != before.round


def test_env_rejects_invalid_index_and_illegal_action() -> None:
    env = TichuEnv(config=_fast_config(), seed=0, dealer=0)
    state = env.state()

    with pytest.raises(IndexError):
        env.step(state.action_count)
    with pytest.raises(ValueError, match="Illegal action"):
        env.step_action(PassAction())


def test_encode_terminal_env_state_uses_team_diff_rewards() -> None:
    round_ = make_round(
        {},
        config=_fast_config(),
        phase=Phase.ROUND_OVER,
        current_player=None,
        finish_order=(0, 2, 1, 3),
        round_scores=(120, -20),
    )

    state = encode_env_state(round_, reward_mode="team_diff", reward_scale=1000.0)

    assert state.terminal
    assert state.observation is None
    assert state.observation_features is None
    assert state.legal_actions == ()
    assert state.encoded_actions == ()
    assert state.rewards == pytest.approx((0.14, -0.14, 0.14, -0.14))


def test_encode_terminal_env_state_can_use_team_score_rewards() -> None:
    round_ = make_round(
        {},
        config=_fast_config(),
        phase=Phase.ROUND_OVER,
        current_player=None,
        finish_order=(0, 2, 1, 3),
        round_scores=(120, -20),
    )

    state = encode_env_state(round_, reward_mode="team_score", reward_scale=1000.0)

    assert state.rewards == pytest.approx((0.12, -0.02, 0.12, -0.02))
