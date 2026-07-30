from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .base import Agent
from .greedy import GreedyAgent
from .heuristic import HeuristicAgent
from .random import RandomAgent


AgentFactory = Callable[[int, str | None], Agent]


@dataclass(frozen=True, slots=True)
class AgentRegistration:
    factory: AgentFactory
    cache_by_spec: bool = False


class AgentCache:
    """Creates agents while reusing registered stateless or checkpoint-backed specs."""

    def __init__(self) -> None:
        self._agents: dict[tuple[str, str | None], Agent] = {}

    def create(self, spec: str, seed: int) -> Agent:
        name, argument = _parse_agent_spec(spec)
        registration = _registration(name)
        if not registration.cache_by_spec:
            return registration.factory(seed, argument)

        key = (name, argument)
        agent = self._agents.get(key)
        if agent is None:
            agent = registration.factory(seed, argument)
            self._agents[key] = agent
        return agent


def _parse_agent_spec(spec: str) -> tuple[str, str | None]:
    name, separator, argument = spec.partition(":")
    if not name:
        raise ValueError("Agent name cannot be empty")
    if separator and not argument:
        raise ValueError(f"Agent spec {spec!r} is missing an argument after ':'")
    return name, argument or None


def _reject_argument(name: str, argument: str | None) -> None:
    if argument is not None:
        raise ValueError(f"Agent {name!r} does not accept an argument")


def _random_agent(seed: int, argument: str | None) -> Agent:
    _reject_argument("random", argument)
    return RandomAgent(seed=seed, bomb_probability=0.15)


def _greedy_agent(seed: int, argument: str | None) -> Agent:
    _ = seed
    _reject_argument("greedy", argument)
    return GreedyAgent()


def _heuristic_agent(seed: int, argument: str | None) -> Agent:
    _ = seed
    _reject_argument("heuristic", argument)
    return HeuristicAgent()


def _actor_critic_agent(seed: int, argument: str | None) -> Agent:
    _ = seed
    from .actor_critic import ActorCriticAgent

    if argument is None:
        raise ValueError("Agent 'actor_critic' requires a checkpoint path")
    return ActorCriticAgent.from_checkpoint(argument, deterministic=True)


_AGENTS: dict[str, AgentRegistration] = {
    "actor_critic": AgentRegistration(_actor_critic_agent, cache_by_spec=True),
    "greedy": AgentRegistration(_greedy_agent, cache_by_spec=True),
    "heuristic": AgentRegistration(_heuristic_agent, cache_by_spec=True),
    "random": AgentRegistration(_random_agent),
}


def available_agent_names() -> tuple[str, ...]:
    return tuple(sorted(_AGENTS))


def create_agent(spec: str, seed: int) -> Agent:
    name, argument = _parse_agent_spec(spec)
    return _registration(name).factory(seed, argument)


def _registration(name: str) -> AgentRegistration:
    try:
        return _AGENTS[name]
    except KeyError as exc:
        available = ", ".join(available_agent_names())
        raise ValueError(
            f"Unknown agent {name!r}. Available agents: {available}. "
            "Use actor_critic:/path/to/checkpoint.pt for a trained ML policy."
        ) from exc
