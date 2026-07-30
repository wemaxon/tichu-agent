from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tichu.actions import Action
from tichu.observations import RoundObservation

from ml.encoding import EncodedAction, encode_legal_actions, encode_observation
from ml.interfaces import EncodedCandidate, InferencePolicy


@dataclass(frozen=True, slots=True)
class ActorCriticAgentState:
    terminal: bool
    observation_features: Sequence[float] | None
    encoded_actions: Sequence[EncodedCandidate]


class ActorCriticAgent:
    """Adapter from a parametric ML policy to the public Agent contract."""

    def __init__(
        self,
        policy: InferencePolicy,
        *,
        deterministic: bool = True,
    ) -> None:
        self.policy = policy
        self.deterministic = deterministic

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        deterministic: bool = True,
    ) -> ActorCriticAgent:
        from ml.policy import ParametricPolicy

        return cls(
            ParametricPolicy(model),
            deterministic=deterministic,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> ActorCriticAgent:
        from ml.checkpoint import load_actor_critic_checkpoint

        return cls.from_model(
            load_actor_critic_checkpoint(path, device=device),
            deterministic=deterministic,
        )

    def choose_action(
        self,
        observation: RoundObservation,
        legal_actions: Sequence[Action],
    ) -> Action:
        if not legal_actions:
            raise RuntimeError("The simulator produced no legal actions")

        player = observation["current_player"]
        if player is None:
            raise RuntimeError("No action is available")

        state = ActorCriticAgentState(
            terminal=False,
            observation_features=encode_observation(observation),
            encoded_actions=encode_legal_actions(legal_actions, observer=player),
        )
        decision = self.policy.decide(
            state,
            deterministic=self.deterministic,
        )
        if decision.action_index < 0 or decision.action_index >= len(legal_actions):
            raise RuntimeError(
                f"Policy selected invalid action index: {decision.action_index}"
            )
        return legal_actions[decision.action_index]
