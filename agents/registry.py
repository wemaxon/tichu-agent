from __future__ import annotations

from collections.abc import Callable

from .base import Agent
from .greedy import GreedyAgent
from .heuristic import HeuristicAgent
from .random import RandomAgent


AgentFactory = Callable[[int, str | None], Agent]


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


_AGENT_FACTORIES: dict[str, AgentFactory] = {
    "actor_critic": _actor_critic_agent,
    "greedy": _greedy_agent,
    "heuristic": _heuristic_agent,
    "random": _random_agent,
}


def available_agent_names() -> tuple[str, ...]:
    return tuple(sorted(_AGENT_FACTORIES))


def create_agent(spec: str, seed: int) -> Agent:
    name, argument = _parse_agent_spec(spec)
    try:
        factory = _AGENT_FACTORIES[name]
    except KeyError as exc:
        available = ", ".join(available_agent_names())
        raise ValueError(
            f"Unknown agent {name!r}. Available agents: {available}. "
            "Use actor_critic:/path/to/checkpoint.pt for a trained ML policy."
        ) from exc
    return factory(seed, argument)
