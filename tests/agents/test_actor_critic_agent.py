from __future__ import annotations

from dataclasses import dataclass

import pytest

from agents.actor_critic import ActorCriticAgent, ActorCriticAgentState
from ml.encoding import ACTION_FEATURE_SIZE, OBSERVATION_FEATURE_SIZE
from ml.interfaces import EncodedPolicyState
from tichu.actions import Action
from tichu.observations import RoundObservation
from tichu.round import TichuRound


@dataclass(frozen=True, slots=True)
class _Decision:
    action_index: int


class _Policy:
    def __init__(self, action_index: int) -> None:
        self.action_index = action_index
        self.seen_state: ActorCriticAgentState | None = None
        self.seen_deterministic: bool | None = None

    def decide(
        self,
        state: EncodedPolicyState,
        *,
        deterministic: bool = False,
    ) -> _Decision:
        assert isinstance(state, ActorCriticAgentState)
        self.seen_state = state
        self.seen_deterministic = deterministic
        return _Decision(self.action_index)


def _turn() -> tuple[int, RoundObservation, tuple[Action, ...]]:
    round_ = TichuRound.new(seed=0, dealer=0)
    player = round_.current_player
    assert player is not None
    return player, round_.observation(player), round_.legal_actions()


def test_actor_critic_agent_selects_policy_action_index() -> None:
    _, observation, actions = _turn()
    policy = _Policy(action_index=1)

    action = ActorCriticAgent(
        policy,
        deterministic=False,
    ).choose_action(observation, actions)

    assert action == actions[1]
    assert policy.seen_deterministic is False
    assert policy.seen_state is not None
    assert policy.seen_state.terminal is False
    assert policy.seen_state.observation_features is not None
    assert len(policy.seen_state.observation_features) == OBSERVATION_FEATURE_SIZE
    assert len(policy.seen_state.encoded_actions) == len(actions)
    assert all(
        len(encoded.features) == ACTION_FEATURE_SIZE
        for encoded in policy.seen_state.encoded_actions
    )


def test_actor_critic_agent_rejects_invalid_policy_action_index() -> None:
    _, observation, actions = _turn()

    with pytest.raises(RuntimeError, match="invalid action index"):
        ActorCriticAgent(_Policy(action_index=len(actions))).choose_action(
            observation,
            actions,
        )


def test_actor_critic_agent_rejects_empty_legal_actions() -> None:
    _, observation, _ = _turn()

    with pytest.raises(RuntimeError, match="no legal actions"):
        ActorCriticAgent(_Policy(action_index=0)).choose_action(observation, ())
