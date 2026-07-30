from __future__ import annotations

import pytest

from agents import AgentCache, available_agent_names, create_agent
from agents.actor_critic import ActorCriticAgent
from agents.greedy import GreedyAgent
from agents.random import RandomAgent
from ml import ACTION_FEATURE_SIZE, OBSERVATION_FEATURE_SIZE
from ml.checkpoint import save_actor_critic_checkpoint
from ml.models import ActorCriticConfig, ParametricActorCritic


def test_agent_registry_lists_actor_critic_base_name() -> None:
    assert "actor_critic" in available_agent_names()


def test_agent_registry_rejects_unknown_agent() -> None:
    with pytest.raises(ValueError, match="Unknown agent"):
        create_agent("unknown", seed=123)


def test_agent_registry_rejects_missing_spec_argument() -> None:
    with pytest.raises(ValueError, match="missing an argument"):
        create_agent("actor_critic:", seed=123)


def test_actor_critic_registry_entry_requires_checkpoint() -> None:
    with pytest.raises(ValueError, match="requires a checkpoint"):
        create_agent("actor_critic", seed=123)


def test_agent_registry_rejects_arguments_for_rule_based_agents() -> None:
    with pytest.raises(ValueError, match="does not accept an argument"):
        create_agent("greedy:model.pt", seed=123)


def test_agent_cache_reuses_cacheable_agents(tmp_path) -> None:
    model = ParametricActorCritic(
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
    checkpoint = tmp_path / "policy.pt"
    save_actor_critic_checkpoint(checkpoint, model)

    cache = AgentCache()

    first_greedy = cache.create("greedy", seed=1)
    second_greedy = cache.create("greedy", seed=2)
    first_policy = cache.create(f"actor_critic:{checkpoint}", seed=1)
    second_policy = cache.create(f"actor_critic:{checkpoint}", seed=2)

    assert isinstance(first_greedy, GreedyAgent)
    assert first_greedy is second_greedy
    assert isinstance(first_policy, ActorCriticAgent)
    assert first_policy is second_policy


def test_agent_cache_keeps_stateful_random_agents_fresh() -> None:
    cache = AgentCache()

    first = cache.create("random", seed=1)
    second = cache.create("random", seed=1)

    assert isinstance(first, RandomAgent)
    assert isinstance(second, RandomAgent)
    assert first is not second
