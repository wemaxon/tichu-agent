from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import torch
from torch import Tensor, nn

from ..batching import CandidateBatch


@dataclass(frozen=True, slots=True)
class ActorCriticConfig:
    observation_size: int
    action_size: int
    state_hidden_size: int = 512
    action_hidden_size: int = 256
    embedding_size: int = 256
    scorer_hidden_size: int = 256
    value_hidden_size: int = 128

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> ActorCriticConfig:
        expected = {field.name for field in fields(cls)}
        actual = set(values)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(
                f"invalid model config keys; missing={missing}, unknown={unknown}"
            )
        return cls(**{name: _positive_int(name, values[name]) for name in expected})


@dataclass(frozen=True, slots=True)
class ActorCriticOutput:
    logits: Tensor
    values: Tensor


class ParametricActorCritic(nn.Module):
    """Scores variable legal actions and estimates one value per state."""

    config: ActorCriticConfig

    def __init__(self, config: ActorCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.state_encoder = _feature_tower(
            config.observation_size,
            config.state_hidden_size,
            config.embedding_size,
        )
        self.action_encoder = _feature_tower(
            config.action_size,
            config.action_hidden_size,
            config.embedding_size,
        )
        self.state_score_projection = nn.Linear(
            config.embedding_size,
            config.scorer_hidden_size,
            bias=False,
        )
        self.action_score_projection = nn.Linear(
            config.embedding_size,
            config.scorer_hidden_size,
            bias=False,
        )
        self.score = nn.Linear(config.scorer_hidden_size, 1)
        self.value_head = nn.Sequential(
            nn.Linear(config.embedding_size, config.value_hidden_size),
            nn.LayerNorm(config.value_hidden_size),
            nn.SiLU(),
            nn.Linear(config.value_hidden_size, 1),
        )

    def forward(self, batch: CandidateBatch) -> ActorCriticOutput:
        if batch.observations.shape[1] != self.config.observation_size:
            raise ValueError(
                f"expected {self.config.observation_size} observation features, "
                f"received {batch.observations.shape[1]}"
            )
        if batch.actions.shape[1] != self.config.action_size:
            raise ValueError(
                f"expected {self.config.action_size} action features, "
                f"received {batch.actions.shape[1]}"
            )

        state_embeddings = self.state_encoder(batch.observations)
        action_embeddings = self.action_encoder(batch.actions)
        candidate_state_scores = self.state_score_projection(
            state_embeddings
        ).index_select(0, batch.candidate_owner)
        candidate_action_scores = self.action_score_projection(action_embeddings)
        logits = self.score(
            torch.tanh(candidate_state_scores + candidate_action_scores)
        ).squeeze(-1)
        values = self.value_head(state_embeddings).squeeze(-1)
        return ActorCriticOutput(logits=logits, values=values)


def _feature_tower(
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, output_size),
        nn.LayerNorm(output_size),
        nn.SiLU(),
    )


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
