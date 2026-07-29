from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from tichu.actions import Action
from tichu.config import TichuRoundConfig
from tichu.observations import RoundObservation
from tichu.round import TichuRound, team_of

from .encoding import (
    EncodedAction,
    FeatureVector,
    encode_legal_actions,
    encode_policy_observation,
)
from .policy_view import PolicyObservation, policy_view

RewardMode: TypeAlias = Literal["team_diff", "team_score"]
PlayerRewards: TypeAlias = tuple[float, float, float, float]
TeamRewards: TypeAlias = tuple[float, float]


@dataclass(frozen=True, slots=True)
class TichuEnvState:
    steps: int
    player: int | None
    observation: RoundObservation | None
    policy_observation: PolicyObservation | None
    observation_features: FeatureVector | None
    legal_actions: tuple[Action, ...]
    encoded_actions: tuple[EncodedAction, ...]
    rewards: PlayerRewards
    terminal: bool

    @property
    def action_count(self) -> int:
        return len(self.legal_actions)


class TichuEnv:
    """Small ML-friendly wrapper around the immutable Tichu round engine."""

    config: TichuRoundConfig
    seed: int
    dealer: int
    reward_mode: RewardMode
    reward_scale: float
    safety_limit: int
    _round: TichuRound
    _steps: int

    def __init__(
        self,
        *,
        config: TichuRoundConfig | None = None,
        seed: int = 0,
        dealer: int = 0,
        reward_mode: RewardMode = "team_diff",
        reward_scale: float = 1000.0,
        safety_limit: int = 20_000,
    ) -> None:
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        if safety_limit < 1:
            raise ValueError("safety_limit must be at least 1")

        self.config = TichuRoundConfig() if config is None else config
        self.seed = int(seed)
        self.dealer = int(dealer)
        self.reward_mode = reward_mode
        self.reward_scale = float(reward_scale)
        self.safety_limit = int(safety_limit)
        self._round = TichuRound.new(seed=self.seed, dealer=self.dealer, config=self.config)
        self._steps = 0

    @property
    def round(self) -> TichuRound:
        return self._round

    @property
    def steps(self) -> int:
        return self._steps

    def reset(
        self,
        *,
        seed: int | None = None,
        dealer: int | None = None,
    ) -> TichuEnvState:
        if seed is not None:
            self.seed = int(seed)
        if dealer is not None:
            self.dealer = int(dealer)

        self._round = TichuRound.new(seed=self.seed, dealer=self.dealer, config=self.config)
        self._steps = 0
        return self.state()

    def state(self) -> TichuEnvState:
        return encode_env_state(
            self._round,
            steps=self._steps,
            reward_mode=self.reward_mode,
            reward_scale=self.reward_scale,
        )

    def step(self, action_index: int) -> TichuEnvState:
        if self._round.terminal:
            raise RuntimeError("Cannot step a terminal environment")

        legal_actions = self._round.legal_actions()
        if action_index < 0 or action_index >= len(legal_actions):
            raise IndexError(f"action_index out of range: {action_index}")
        return self.step_action(legal_actions[action_index])

    def step_action(self, action: Action) -> TichuEnvState:
        if self._round.terminal:
            raise RuntimeError("Cannot step a terminal environment")
        if action not in self._round.legal_actions():
            raise ValueError(f"Illegal action: {action!r}")

        self._round = self._round.apply(action)
        self._steps += 1
        if self._steps > self.safety_limit:
            raise RuntimeError("Environment exceeded the safety step limit")
        return self.state()


def encode_env_state(
    round_: TichuRound,
    *,
    steps: int = 0,
    reward_mode: RewardMode = "team_diff",
    reward_scale: float = 1000.0,
) -> TichuEnvState:
    if reward_scale <= 0:
        raise ValueError("reward_scale must be positive")

    if round_.terminal:
        return TichuEnvState(
            steps=steps,
            player=None,
            observation=None,
            policy_observation=None,
            observation_features=None,
            legal_actions=(),
            encoded_actions=(),
            rewards=_terminal_rewards(round_, reward_mode=reward_mode, reward_scale=reward_scale),
            terminal=True,
        )

    player = round_.current_player
    if player is None:
        raise RuntimeError("Non-terminal round has no current player")

    observation = round_.observation(player)
    policy_observation = policy_view(observation)
    legal_actions = round_.legal_actions()
    return TichuEnvState(
        steps=steps,
        player=player,
        observation=observation,
        policy_observation=policy_observation,
        observation_features=encode_policy_observation(policy_observation),
        legal_actions=legal_actions,
        encoded_actions=encode_legal_actions(legal_actions, observer=player),
        rewards=(0.0, 0.0, 0.0, 0.0),
        terminal=False,
    )


def _terminal_rewards(
    round_: TichuRound,
    *,
    reward_mode: RewardMode,
    reward_scale: float,
) -> PlayerRewards:
    if round_.round_scores is None:
        raise RuntimeError("Terminal round has no scores")

    score_0, score_1 = round_.round_scores
    if reward_mode == "team_score":
        team_rewards: TeamRewards = (score_0 / reward_scale, score_1 / reward_scale)
    elif reward_mode == "team_diff":
        diff = (score_0 - score_1) / reward_scale
        team_rewards = (diff, -diff)
    else:
        raise ValueError(f"Unsupported reward_mode: {reward_mode!r}")

    return (
        team_rewards[team_of(0)],
        team_rewards[team_of(1)],
        team_rewards[team_of(2)],
        team_rewards[team_of(3)],
    )
