from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import statistics
from typing import Protocol

import torch
from torch import Tensor

from tichu.actions import Action
from tichu.config import TichuRoundConfig
from tichu.observations import RoundObservation
from tichu.round import team_of

from .batching import batch_env_states
from .distributions import RaggedCategorical
from .env import RewardMode, TichuEnv, TichuEnvState
from .models import ParametricActorCritic


class BaselineAgent(Protocol):
    def choose_action(
        self,
        observation: RoundObservation,
        legal_actions: Sequence[Action],
    ) -> Action:
        ...


@dataclass(frozen=True, slots=True)
class ActorCriticTrainingConfig:
    learner_players: tuple[int, ...] = (0, 2)
    reward_mode: RewardMode = "team_diff"
    reward_scale: float = 1000.0
    value_loss_weight: float = 0.5
    entropy_bonus_weight: float = 0.01
    max_grad_norm: float | None = 1.0
    safety_limit: int = 20_000

    def __post_init__(self) -> None:
        _validate_players(self.learner_players, "learner_players")
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        if self.value_loss_weight < 0:
            raise ValueError("value_loss_weight cannot be negative")
        if self.entropy_bonus_weight < 0:
            raise ValueError("entropy_bonus_weight cannot be negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        if self.safety_limit < 1:
            raise ValueError("safety_limit must be at least 1")


@dataclass(frozen=True, slots=True)
class LearnerTransition:
    player: int
    action_index: int
    log_probability: Tensor
    value: Tensor
    entropy: Tensor


@dataclass(frozen=True, slots=True)
class TrainingRollout:
    seed: int
    dealer: int
    steps: int
    transitions: tuple[LearnerTransition, ...]
    rewards: tuple[float, float, float, float]
    team_scores: tuple[int, int]

    @property
    def learner_steps(self) -> int:
        return len(self.transitions)


@dataclass(frozen=True, slots=True)
class ActorCriticLoss:
    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    mean_return: Tensor


@dataclass(frozen=True, slots=True)
class TrainingRoundResult:
    seed: int
    dealer: int
    steps: int
    learner_steps: int
    rewards: tuple[float, float, float, float]
    team_scores: tuple[int, int]
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    mean_return: float
    grad_norm: float | None
    updated: bool


def learner_players_for_team(team: int) -> tuple[int, int]:
    if team not in (0, 1):
        raise ValueError(f"team must be 0 or 1: {team}")
    players = tuple(player for player in range(4) if team_of(player) == team)
    return players[0], players[1]


def collect_training_round(
    model: ParametricActorCritic,
    baseline_agents: Mapping[int, BaselineAgent],
    *,
    seed: int,
    dealer: int,
    round_config: TichuRoundConfig | None = None,
    training_config: ActorCriticTrainingConfig | None = None,
    generator: torch.Generator | None = None,
) -> TrainingRollout:
    settings = ActorCriticTrainingConfig() if training_config is None else training_config
    learner_players = set(settings.learner_players)
    _validate_baseline_agents(baseline_agents, learner_players)

    model.train()
    env = TichuEnv(
        config=round_config,
        seed=seed,
        dealer=dealer,
        reward_mode=settings.reward_mode,
        reward_scale=settings.reward_scale,
        safety_limit=settings.safety_limit,
    )
    state = env.state()
    transitions: list[LearnerTransition] = []

    while not state.terminal:
        player = state.player
        if player is None:
            raise RuntimeError("Non-terminal training state has no current player")

        if player in learner_players:
            transition = _sample_learner_transition(
                model,
                state,
                player=player,
                generator=generator,
            )
            transitions.append(transition)
            state = env.step(transition.action_index)
        else:
            action = _baseline_action(baseline_agents[player], state)
            state = env.step_action(action)

    team_scores = env.round.round_scores
    if team_scores is None:
        raise RuntimeError("Terminal training round has no team scores")

    return TrainingRollout(
        seed=seed,
        dealer=dealer,
        steps=state.steps,
        transitions=tuple(transitions),
        rewards=state.rewards,
        team_scores=team_scores,
    )


def compute_actor_critic_loss(
    rollout: TrainingRollout,
    *,
    training_config: ActorCriticTrainingConfig | None = None,
) -> ActorCriticLoss:
    settings = ActorCriticTrainingConfig() if training_config is None else training_config
    if not rollout.transitions:
        return _zero_loss()

    values = torch.stack([transition.value for transition in rollout.transitions])
    log_probabilities = torch.stack(
        [transition.log_probability for transition in rollout.transitions]
    )
    entropies = torch.stack([transition.entropy for transition in rollout.transitions])
    returns = torch.tensor(
        [rollout.rewards[transition.player] for transition in rollout.transitions],
        dtype=values.dtype,
        device=values.device,
    )
    advantages = returns - values.detach()

    policy_loss = -(log_probabilities * advantages).mean()
    value_loss = (values - returns).pow(2).mean()
    entropy = entropies.mean()
    total_loss = (
        policy_loss
        + settings.value_loss_weight * value_loss
        - settings.entropy_bonus_weight * entropy
    )
    return ActorCriticLoss(
        total=total_loss,
        policy=policy_loss,
        value=value_loss,
        entropy=entropy,
        mean_return=returns.mean(),
    )


def train_actor_critic_round(
    model: ParametricActorCritic,
    optimizer: torch.optim.Optimizer,
    baseline_agents: Mapping[int, BaselineAgent],
    *,
    seed: int,
    dealer: int,
    round_config: TichuRoundConfig | None = None,
    training_config: ActorCriticTrainingConfig | None = None,
    generator: torch.Generator | None = None,
) -> TrainingRoundResult:
    settings = ActorCriticTrainingConfig() if training_config is None else training_config
    rollout = collect_training_round(
        model,
        baseline_agents,
        seed=seed,
        dealer=dealer,
        round_config=round_config,
        training_config=settings,
        generator=generator,
    )
    loss = compute_actor_critic_loss(rollout, training_config=settings)

    grad_norm: float | None = None
    updated = bool(rollout.transitions)
    if updated:
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        if settings.max_grad_norm is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    settings.max_grad_norm,
                ).detach().cpu().item()
            )
        optimizer.step()

    return TrainingRoundResult(
        seed=rollout.seed,
        dealer=rollout.dealer,
        steps=rollout.steps,
        learner_steps=rollout.learner_steps,
        rewards=rollout.rewards,
        team_scores=rollout.team_scores,
        total_loss=_scalar(loss.total),
        policy_loss=_scalar(loss.policy),
        value_loss=_scalar(loss.value),
        entropy=_scalar(loss.entropy),
        mean_return=_scalar(loss.mean_return),
        grad_norm=grad_norm,
        updated=updated,
    )


def mean_training_result(results: Sequence[TrainingRoundResult]) -> dict[str, float]:
    if not results:
        raise ValueError("at least one result is required")
    return {
        "total_loss": statistics.fmean(result.total_loss for result in results),
        "policy_loss": statistics.fmean(result.policy_loss for result in results),
        "value_loss": statistics.fmean(result.value_loss for result in results),
        "entropy": statistics.fmean(result.entropy for result in results),
        "mean_return": statistics.fmean(result.mean_return for result in results),
        "steps": statistics.fmean(result.steps for result in results),
        "learner_steps": statistics.fmean(result.learner_steps for result in results),
    }


def _sample_learner_transition(
    model: ParametricActorCritic,
    state: TichuEnvState,
    *,
    player: int,
    generator: torch.Generator | None,
) -> LearnerTransition:
    parameter = next(model.parameters())
    batch = batch_env_states(
        (state,),
        device=parameter.device,
        dtype=parameter.dtype,
    )
    output = model(batch)
    distribution = RaggedCategorical(output.logits, batch.offsets)
    local_indices = distribution.sample(generator=generator)
    log_probability = distribution.log_prob(local_indices)[0]
    return LearnerTransition(
        player=player,
        action_index=int(local_indices[0].detach().cpu().item()),
        log_probability=log_probability,
        value=output.values[0],
        entropy=distribution.entropy()[0],
    )


def _baseline_action(agent: BaselineAgent, state: TichuEnvState) -> Action:
    if state.observation is None:
        raise RuntimeError("Non-terminal baseline state has no observation")
    action = agent.choose_action(state.observation, state.legal_actions)
    if action not in state.legal_actions:
        raise RuntimeError(f"Baseline agent chose an illegal action: {action!r}")
    return action


def _validate_baseline_agents(
    baseline_agents: Mapping[int, BaselineAgent],
    learner_players: set[int],
) -> None:
    for player in baseline_agents:
        if player not in range(4):
            raise ValueError(f"baseline player must be between 0 and 3: {player}")
        if player in learner_players:
            raise ValueError(f"player {player} cannot be both learner and baseline")
    missing = [
        player
        for player in range(4)
        if player not in learner_players and player not in baseline_agents
    ]
    if missing:
        raise ValueError(f"Missing baseline agents for players: {missing}")


def _validate_players(players: tuple[int, ...], name: str) -> None:
    if not players:
        raise ValueError(f"{name} cannot be empty")
    if len(set(players)) != len(players):
        raise ValueError(f"{name} cannot contain duplicates")
    for player in players:
        if player not in range(4):
            raise ValueError(f"{name} must contain players between 0 and 3: {player}")


def _zero_loss() -> ActorCriticLoss:
    zero = torch.zeros(())
    return ActorCriticLoss(
        total=zero,
        policy=zero,
        value=zero,
        entropy=zero,
        mean_return=zero,
    )


def _scalar(value: Tensor) -> float:
    return float(value.detach().cpu().item())
