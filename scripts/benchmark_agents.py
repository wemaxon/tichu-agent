from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import Agent, available_agent_names, create_agent
from scripts.round_runner import add_round_config_args, play_agent_round, round_config_from_args
from tichu.config import TichuRoundConfig
from tichu.round import team_of


AgentLineup = tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class PlayedRound:
    index: int
    seed: int
    dealer: int
    lineup: AgentLineup
    agent_a_team: int
    team_scores: tuple[int, int]
    finish_order: tuple[int, ...]
    steps: int

    @property
    def agent_a_score(self) -> int:
        return self.team_scores[self.agent_a_team]

    @property
    def agent_b_score(self) -> int:
        return self.team_scores[1 - self.agent_a_team]

    @property
    def score_diff(self) -> int:
        return self.agent_a_score - self.agent_b_score

    @property
    def double_victory_team(self) -> int | None:
        if (
            len(self.finish_order) >= 2
            and team_of(self.finish_order[0]) == team_of(self.finish_order[1])
        ):
            return team_of(self.finish_order[0])
        return None

    @property
    def double_victory_for(self) -> str | None:
        team = self.double_victory_team
        if team is None:
            return None
        return "A" if team == self.agent_a_team else "B"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "seed": self.seed,
            "dealer": self.dealer,
            "lineup": list(self.lineup),
            "agent_a_team": self.agent_a_team,
            "team_scores": list(self.team_scores),
            "agent_a_score": self.agent_a_score,
            "agent_b_score": self.agent_b_score,
            "score_diff": self.score_diff,
            "finish_order": list(self.finish_order),
            "steps": self.steps,
            "double_victory_for": self.double_victory_for,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    agent_a: str
    agent_b: str
    round_pairs: int
    seed: int
    dealer: int
    config: TichuRoundConfig
    rounds: tuple[PlayedRound, ...]
    elapsed_seconds: float

    @property
    def played_rounds(self) -> int:
        return len(self.rounds)

    @property
    def agent_a_score(self) -> int:
        return sum(round_.agent_a_score for round_ in self.rounds)

    @property
    def agent_b_score(self) -> int:
        return sum(round_.agent_b_score for round_ in self.rounds)

    @property
    def score_diff(self) -> int:
        return self.agent_a_score - self.agent_b_score

    @property
    def score_diffs(self) -> tuple[int, ...]:
        return tuple(round_.score_diff for round_ in self.rounds)

    @property
    def agent_a_wins(self) -> int:
        return sum(1 for round_ in self.rounds if round_.agent_a_score > round_.agent_b_score)

    @property
    def agent_b_wins(self) -> int:
        return sum(1 for round_ in self.rounds if round_.agent_b_score > round_.agent_a_score)

    @property
    def ties(self) -> int:
        return self.played_rounds - self.agent_a_wins - self.agent_b_wins

    @property
    def agent_a_double_victories(self) -> int:
        return sum(1 for round_ in self.rounds if round_.double_victory_for == "A")

    @property
    def agent_b_double_victories(self) -> int:
        return sum(1 for round_ in self.rounds if round_.double_victory_for == "B")

    @property
    def mean_score_diff(self) -> float:
        if not self.rounds:
            return 0.0
        return statistics.fmean(self.score_diffs)

    @property
    def stdev_score_diff(self) -> float:
        if len(self.rounds) < 2:
            return 0.0
        return statistics.stdev(self.score_diffs)

    @property
    def mean_steps(self) -> float:
        if not self.rounds:
            return 0.0
        return statistics.fmean(round_.steps for round_ in self.rounds)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "round_pairs": self.round_pairs,
            "played_rounds": self.played_rounds,
            "seed": self.seed,
            "dealer": self.dealer,
            "config": self.config.to_dict(),
            "elapsed_seconds": self.elapsed_seconds,
            "agent_a_score": self.agent_a_score,
            "agent_b_score": self.agent_b_score,
            "score_diff": self.score_diff,
            "mean_score_diff": self.mean_score_diff,
            "stdev_score_diff": self.stdev_score_diff,
            "agent_a_wins": self.agent_a_wins,
            "agent_b_wins": self.agent_b_wins,
            "ties": self.ties,
            "agent_a_double_victories": self.agent_a_double_victories,
            "agent_b_double_victories": self.agent_b_double_victories,
            "mean_steps": self.mean_steps,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary_dict(),
            "rounds": [round_.to_dict() for round_ in self.rounds],
        }


def run_benchmark(
    *,
    agent_a: str,
    agent_b: str,
    round_pairs: int,
    seed: int,
    dealer: int,
    config: TichuRoundConfig,
) -> BenchmarkResult:
    if round_pairs < 1:
        raise ValueError("round_pairs must be at least 1")

    started = time.perf_counter()
    rounds: list[PlayedRound] = []
    for pair_index in range(round_pairs):
        round_seed = seed + pair_index
        round_dealer = (dealer + pair_index) % 4
        rounds.append(
            _play_benchmark_round(
                index=len(rounds),
                seed=round_seed,
                dealer=round_dealer,
                lineup=(agent_a, agent_b, agent_a, agent_b),
                agent_a_team=0,
                config=config,
            )
        )
        rounds.append(
            _play_benchmark_round(
                index=len(rounds),
                seed=round_seed,
                dealer=round_dealer,
                lineup=(agent_b, agent_a, agent_b, agent_a),
                agent_a_team=1,
                config=config,
            )
        )

    return BenchmarkResult(
        agent_a=agent_a,
        agent_b=agent_b,
        round_pairs=round_pairs,
        seed=seed,
        dealer=dealer,
        config=config,
        rounds=tuple(rounds),
        elapsed_seconds=time.perf_counter() - started,
    )


def _play_benchmark_round(
    *,
    index: int,
    seed: int,
    dealer: int,
    lineup: AgentLineup,
    agent_a_team: int,
    config: TichuRoundConfig,
) -> PlayedRound:
    agents = _build_agents(lineup, seed=seed, offset=index)
    result = play_agent_round(seed=seed, dealer=dealer, agents=agents, config=config)
    round_ = result.round

    if round_.round_scores is None:
        raise RuntimeError("Terminal round has no score")

    return PlayedRound(
        index=index,
        seed=seed,
        dealer=dealer,
        lineup=lineup,
        agent_a_team=agent_a_team,
        team_scores=round_.round_scores,
        finish_order=round_.finish_order,
        steps=result.steps,
    )


def _build_agents(lineup: AgentLineup, *, seed: int, offset: int) -> list[Agent]:
    return [
        create_agent(agent_name, seed * 10_000 + offset * 100 + player)
        for player, agent_name in enumerate(lineup)
    ]


def write_json_result(result: BenchmarkResult, path: str) -> None:
    Path(path).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def write_csv_rounds(result: BenchmarkResult, path: str) -> None:
    fieldnames = [
        "index",
        "seed",
        "dealer",
        "lineup",
        "agent_a_team",
        "team0_score",
        "team1_score",
        "agent_a_score",
        "agent_b_score",
        "score_diff",
        "finish_order",
        "steps",
        "double_victory_for",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for round_ in result.rounds:
            writer.writerow(
                {
                    "index": round_.index,
                    "seed": round_.seed,
                    "dealer": round_.dealer,
                    "lineup": " ".join(round_.lineup),
                    "agent_a_team": round_.agent_a_team,
                    "team0_score": round_.team_scores[0],
                    "team1_score": round_.team_scores[1],
                    "agent_a_score": round_.agent_a_score,
                    "agent_b_score": round_.agent_b_score,
                    "score_diff": round_.score_diff,
                    "finish_order": " ".join(str(player) for player in round_.finish_order),
                    "steps": round_.steps,
                    "double_victory_for": round_.double_victory_for or "",
                }
            )


def print_summary(result: BenchmarkResult) -> None:
    summary = result.summary_dict()
    win_rate = 100.0 * result.agent_a_wins / result.played_rounds
    print(f"Benchmark: {result.agent_a}+{result.agent_a} vs {result.agent_b}+{result.agent_b}")
    print(f"Round pairs: {result.round_pairs} ({result.played_rounds} played rounds)")
    print(f"Seed: {result.seed} | Dealer start: P{result.dealer}")
    print(f"Score: A={result.agent_a_score} B={result.agent_b_score} diff={result.score_diff:+d}")
    print(
        f"Avg diff/round: {result.mean_score_diff:+.2f} "
        f"(stdev {result.stdev_score_diff:.2f})"
    )
    print(
        f"Wins: A={result.agent_a_wins} B={result.agent_b_wins} "
        f"ties={result.ties} | A win rate={win_rate:.1f}%"
    )
    print(
        "Double victories: "
        f"A={summary['agent_a_double_victories']} B={summary['agent_b_double_victories']}"
    )
    print(f"Avg steps/round: {result.mean_steps:.1f}")
    print(f"Elapsed: {result.elapsed_seconds:.2f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark two Tichu agents with paired rounds")
    parser.add_argument("--agent-a", choices=available_agent_names(), required=True)
    parser.add_argument("--agent-b", choices=available_agent_names(), required=True)
    parser.add_argument(
        "--round-pairs",
        type=int,
        default=100,
        help="Number of paired seeds to run; each pair plays two rounds with teams swapped",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dealer", type=int, choices=range(4), default=0)
    parser.add_argument("--json", type=str, help="Write summary and per-round results as JSON")
    parser.add_argument("--csv", type=str, help="Write per-round results as CSV")
    add_round_config_args(parser)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = run_benchmark(
        agent_a=args.agent_a,
        agent_b=args.agent_b,
        round_pairs=args.round_pairs,
        seed=args.seed,
        dealer=args.dealer,
        config=round_config_from_args(args),
    )
    print_summary(result)
    if args.json:
        write_json_result(result, args.json)
        print(f"JSON written to {args.json}")
    if args.csv:
        write_csv_rounds(result, args.csv)
        print(f"CSV written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
