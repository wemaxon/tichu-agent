from __future__ import annotations

from dataclasses import dataclass

import torch

from .batching import EncodedPolicyState, batch_env_states
from .distributions import RaggedCategorical
from .encoding import ACTION_FEATURE_SIZE, OBSERVATION_FEATURE_SIZE
from .models import ActorCriticConfig, ParametricActorCritic


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action_index: int
    log_probability: float
    value: float
    entropy: float


def default_actor_critic_config() -> ActorCriticConfig:
    return ActorCriticConfig(
        observation_size=OBSERVATION_FEATURE_SIZE,
        action_size=ACTION_FEATURE_SIZE,
    )


def create_actor_critic(
    config: ActorCriticConfig | None = None,
    *,
    device: torch.device | str | None = None,
) -> ParametricActorCritic:
    model = ParametricActorCritic(
        default_actor_critic_config() if config is None else config
    )
    if device is not None:
        model.to(device)
    return model


class ParametricPolicy:
    """Inference adapter from an encoded environment state to a legal action index."""

    def __init__(self, model: ParametricActorCritic) -> None:
        _validate_current_feature_sizes(model.config)
        self.model = model

    @torch.inference_mode()
    def decide(
        self,
        state: EncodedPolicyState,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> PolicyDecision:
        parameter = next(self.model.parameters())
        batch = batch_env_states(
            (state,),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        output = self.model(batch)
        distribution = RaggedCategorical(output.logits, batch.offsets)
        local_indices = (
            distribution.mode()
            if deterministic
            else distribution.sample(generator=generator)
        )
        log_probability = distribution.log_prob(local_indices)[0]
        return PolicyDecision(
            action_index=int(local_indices[0].item()),
            log_probability=float(log_probability.item()),
            value=float(output.values[0].item()),
            entropy=float(distribution.entropy()[0].item()),
        )


def _validate_current_feature_sizes(config: ActorCriticConfig) -> None:
    if config.observation_size != OBSERVATION_FEATURE_SIZE:
        raise ValueError(
            f"model expects {config.observation_size} observation features; "
            f"the current encoder produces {OBSERVATION_FEATURE_SIZE}"
        )
    if config.action_size != ACTION_FEATURE_SIZE:
        raise ValueError(
            f"model expects {config.action_size} action features; "
            f"the current encoder produces {ACTION_FEATURE_SIZE}"
        )
