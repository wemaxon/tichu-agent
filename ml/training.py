from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import statistics
from typing import Literal, Protocol

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


AdvantageNormalization = Literal["none", "batch", "rollout"]


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
    advantage_normalization: AdvantageNormalization = "batch"
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
        if self.advantage_normalization not in ("none", "batch", "rollout"):
            raise ValueError(
                "advantage_normalization must be 'none', 'batch', or 'rollout'"
            )
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
class TrainingBatch:
    rollouts: tuple[TrainingRollout, ...]

    def __post_init__(self) -> None:
        if not self.rollouts:
            raise ValueError("TrainingBatch requires at least one rollout")

    @property
    def round_count(self) -> int:
        return len(self.rollouts)

    @property
    def steps(self) -> int:
        return sum(rollout.steps for rollout in self.rollouts)

    @property
    def learner_steps(self) -> int:
        return sum(rollout.learner_steps for rollout in self.rollouts)

    @property
    def transitions(self) -> tuple[LearnerTransition, ...]:
        return tuple(
            transition
            for rollout in self.rollouts
            for transition in rollout.transitions
        )

    @property
    def first_seed(self) -> int:
        return self.rollouts[0].seed

    @property
    def last_seed(self) -> int:
        return self.rollouts[-1].seed

    @property
    def first_dealer(self) -> int:
        return self.rollouts[0].dealer

    @property
    def last_dealer(self) -> int:
        return self.rollouts[-1].dealer

    @property
    def last_team_scores(self) -> tuple[int, int]:
        return self.rollouts[-1].team_scores


@dataclass(frozen=True, slots=True)
class ActorCriticLoss:
    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    mean_return: Tensor


@dataclass(frozen=True, slots=True)
class TrainingBatchResult:
    first_seed: int
    last_seed: int
    first_dealer: int
    last_dealer: int
    round_count: int
    steps: int
    learner_steps: int
    mean_reward: float
    mean_score_diff: float
    last_team_scores: tuple[int, int]
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    mean_return: float
    grad_norm: float | None
    updated: bool
    entropy_bonus_weight: float
    advantage_normalization: AdvantageNormalization


def learner_players_for_team(team: int) -> tuple[int, int]:
    if team not in (0, 1):
        raise ValueError(f"team must be 0 or 1: {team}")
    players = tuple(player for player in range(4) if team_of(player) == team)
    return players[0], players[1]


def collect_training_batch(
    model: ParametricActorCritic,
    baseline_agents: Mapping[int, BaselineAgent],
    *,
    seeds: Sequence[int],
    dealers: Sequence[int],
    round_config: TichuRoundConfig | None = None,
    training_config: ActorCriticTrainingConfig | None = None,
    generator: torch.Generator | None = None,
) -> TrainingBatch:
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if len(seeds) != len(dealers):
        raise ValueError("seeds and dealers must have the same length")

    settings = ActorCriticTrainingConfig() if training_config is None else training_config
    rollouts = tuple(
        collect_training_round(
            model,
            baseline_agents,
            seed=seed,
            dealer=dealer,
            round_config=round_config,
            training_config=settings,
            generator=generator,
        )
        for seed, dealer in zip(seeds, dealers, strict=True)
    )
    return TrainingBatch(rollouts)


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
            state = env.step(transition.action_index, legal_actions=state.legal_actions)
        else:
            action = _baseline_action(baseline_agents[player], state)
            state = env.step_action(action, legal_actions=state.legal_actions)

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
    batch: TrainingBatch,
    *,
    training_config: ActorCriticTrainingConfig | None = None,
) -> ActorCriticLoss:
    settings = ActorCriticTrainingConfig() if training_config is None else training_config
    transitions = batch.transitions
    if not transitions:
        return _zero_loss()

    values = torch.stack([transition.value for transition in transitions])
    log_probabilities = torch.stack(
        [transition.log_probability for transition in transitions]
    )
    entropies = torch.stack([transition.entropy for transition in transitions])
    returns = _batch_returns(batch, dtype=values.dtype, device=values.device)
    advantages = _normalized_advantages(
        batch,
        returns - values.detach(),
        mode=settings.advantage_normalization,
    )

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


def train_actor_critic_batch(
    model: ParametricActorCritic,
    optimizer: torch.optim.Optimizer,
    baseline_agents: Mapping[int, BaselineAgent],
    *,
    seeds: Sequence[int],
    dealers: Sequence[int],
    round_config: TichuRoundConfig | None = None,
    training_config: ActorCriticTrainingConfig | None = None,
    generator: torch.Generator | None = None,
) -> TrainingBatchResult:
    settings = ActorCriticTrainingConfig() if training_config is None else training_config
    batch = collect_training_batch(
        model,
        baseline_agents,
        seeds=seeds,
        dealers=dealers,
        round_config=round_config,
        training_config=settings,
        generator=generator,
    )
    loss = compute_actor_critic_loss(batch, training_config=settings)

    grad_norm: float | None = None
    updated = bool(batch.learner_steps)
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

    learner_team = _learner_team(settings.learner_players)
    return TrainingBatchResult(
        first_seed=batch.first_seed,
        last_seed=batch.last_seed,
        first_dealer=batch.first_dealer,
        last_dealer=batch.last_dealer,
        round_count=batch.round_count,
        steps=batch.steps,
        learner_steps=batch.learner_steps,
        mean_reward=_mean_learner_reward(batch.rollouts, settings.learner_players),
        mean_score_diff=_mean_score_diff(batch.rollouts, learner_team),
        last_team_scores=batch.last_team_scores,
        total_loss=_scalar(loss.total),
        policy_loss=_scalar(loss.policy),
        value_loss=_scalar(loss.value),
        entropy=_scalar(loss.entropy),
        mean_return=_scalar(loss.mean_return),
        grad_norm=grad_norm,
        updated=updated,
        entropy_bonus_weight=settings.entropy_bonus_weight,
        advantage_normalization=settings.advantage_normalization,
    )


def summarize_training_batches(results: Sequence[TrainingBatchResult]) -> dict[str, float]:
    if not results:
        raise ValueError("at least one result is required")
    learner_steps = sum(result.learner_steps for result in results)
    round_count = sum(result.round_count for result in results)
    grad_norms = [
        result.grad_norm for result in results if result.grad_norm is not None
    ]
    return {
        "total_loss": _weighted_mean(
            (result.total_loss, result.learner_steps) for result in results
        ),
        "policy_loss": _weighted_mean(
            (result.policy_loss, result.learner_steps) for result in results
        ),
        "value_loss": _weighted_mean(
            (result.value_loss, result.learner_steps) for result in results
        ),
        "entropy": _weighted_mean(
            (result.entropy, result.learner_steps) for result in results
        ),
        "mean_return": _weighted_mean(
            (result.mean_return, result.learner_steps) for result in results
        ),
        "mean_reward": _weighted_mean(
            (result.mean_reward, result.round_count) for result in results
        ),
        "mean_score_diff": _weighted_mean(
            (result.mean_score_diff, result.round_count) for result in results
        ),
        "steps": sum(result.steps for result in results) / round_count,
        "learner_steps": learner_steps / round_count,
        "round_count": float(round_count),
        "optimizer_updates": float(len(results)),
        "grad_norm": statistics.fmean(grad_norms) if grad_norms else 0.0,
        "entropy_bonus_weight": results[-1].entropy_bonus_weight,
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


def _batch_returns(
    batch: TrainingBatch,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    return torch.tensor(
        [
            rollout.rewards[transition.player]
            for rollout in batch.rollouts
            for transition in rollout.transitions
        ],
        dtype=dtype,
        device=device,
    )


def _normalized_advantages(
    batch: TrainingBatch,
    advantages: Tensor,
    *,
    mode: AdvantageNormalization,
) -> Tensor:
    if mode == "none":
        return advantages
    if mode == "batch":
        return _normalize_tensor(advantages)

    normalized: list[Tensor] = []
    offset = 0
    for rollout in batch.rollouts:
        count = rollout.learner_steps
        if count:
            normalized.append(_normalize_tensor(advantages[offset : offset + count]))
            offset += count
    if not normalized:
        return advantages
    return torch.cat(normalized)


def _normalize_tensor(values: Tensor) -> Tensor:
    if values.numel() < 2:
        return values - values.mean()
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)


def _learner_team(learner_players: tuple[int, ...]) -> int:
    teams = {team_of(player) for player in learner_players}
    if len(teams) != 1:
        raise ValueError("learner_players must all belong to the same team")
    return teams.pop()


def _mean_learner_reward(
    rollouts: Sequence[TrainingRollout],
    learner_players: tuple[int, ...],
) -> float:
    return statistics.fmean(
        rollout.rewards[player]
        for rollout in rollouts
        for player in learner_players
    )


def _mean_score_diff(rollouts: Sequence[TrainingRollout], learner_team: int) -> float:
    opponent_team = 1 - learner_team
    return statistics.fmean(
        rollout.team_scores[learner_team] - rollout.team_scores[opponent_team]
        for rollout in rollouts
    )


def _weighted_mean(values: Iterable[tuple[float, int]]) -> float:
    total = 0.0
    weight_sum = 0
    fallback: list[float] = []
    for value, weight in values:
        fallback.append(value)
        if weight > 0:
            total += value * weight
            weight_sum += weight
    if weight_sum:
        return total / weight_sum
    return statistics.fmean(fallback)


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
