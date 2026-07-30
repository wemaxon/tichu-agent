from __future__ import annotations

import pytest

from agents import available_agent_names, create_agent


def test_agent_registry_lists_actor_critic_base_name() -> None:
    assert "actor_critic" in available_agent_names()


def test_agent_registry_rejects_unknown_agent() -> None:
    with pytest.raises(ValueError, match="Unknown agent"):
        create_agent("unknown", seed=123)


def test_agent_registry_rejects_missing_spec_argument() -> None:
    with pytest.raises(ValueError, match="missing an argument"):
        create_agent("actor_critic:", seed=123)


def test_agent_registry_rejects_arguments_for_rule_based_agents() -> None:
    with pytest.raises(ValueError, match="does not accept an argument"):
        create_agent("greedy:model.pt", seed=123)
