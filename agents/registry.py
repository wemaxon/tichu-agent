from __future__ import annotations

from collections.abc import Callable

from .base import Agent
from .greedy import GreedyAgent
from .random import RandomAgent


AgentFactory = Callable[[int], Agent]


def _random_agent(seed: int) -> Agent:
    return RandomAgent(seed=seed, bomb_probability=0.15)


def _greedy_agent(seed: int) -> Agent:
    _ = seed
    return GreedyAgent()


_AGENT_FACTORIES: dict[str, AgentFactory] = {
    "greedy": _greedy_agent,
    "random": _random_agent,
}


def available_agent_names() -> tuple[str, ...]:
    return tuple(sorted(_AGENT_FACTORIES))


def create_agent(name: str, seed: int) -> Agent:
    try:
        factory = _AGENT_FACTORIES[name]
    except KeyError as exc:
        available = ", ".join(available_agent_names())
        raise ValueError(f"Unknown agent {name!r}. Available agents: {available}") from exc
    return factory(seed)
