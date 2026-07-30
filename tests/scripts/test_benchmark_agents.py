"""Tests for the paired agent benchmark CLI."""

from __future__ import annotations

import csv
import io
import json

from scripts.benchmark_agents import (
    ProgressBar,
    run_benchmark,
    write_csv_rounds,
    write_json_result,
)
from tichu.config import TichuRoundConfig


def _fast_config() -> TichuRoundConfig:
    return TichuRoundConfig(
        grand_tichu=False,
        small_tichu=True,
        card_exchange=False,
        bombs=True,
        wishes=False,
        special_cards=False,
    )


def test_benchmark_runs_paired_rounds_and_swaps_teams() -> None:
    result = run_benchmark(
        agent_a="greedy",
        agent_b="random",
        round_pairs=1,
        seed=0,
        dealer=0,
        config=_fast_config(),
    )

    assert result.round_pairs == 1
    assert result.played_rounds == 2
    assert tuple(round_.agent_a_team for round_ in result.rounds) == (0, 1)
    assert result.rounds[0].lineup == ("greedy", "random", "greedy", "random")
    assert result.rounds[1].lineup == ("random", "greedy", "random", "greedy")
    assert result.agent_a_score == sum(round_.agent_a_score for round_ in result.rounds)
    assert result.agent_b_score == sum(round_.agent_b_score for round_ in result.rounds)
    assert result.score_diff == result.agent_a_score - result.agent_b_score
    assert all(round_.steps > 0 for round_ in result.rounds)


def test_benchmark_writes_json_and_csv(tmp_path) -> None:
    result = run_benchmark(
        agent_a="greedy",
        agent_b="random",
        round_pairs=1,
        seed=1,
        dealer=2,
        config=_fast_config(),
    )
    json_path = tmp_path / "benchmark.json"
    csv_path = tmp_path / "benchmark.csv"

    write_json_result(result, str(json_path))
    write_csv_rounds(result, str(csv_path))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["agent_a"] == "greedy"
    assert payload["summary"]["agent_b"] == "random"
    assert payload["summary"]["played_rounds"] == 2
    assert len(payload["rounds"]) == 2

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["lineup"] == "greedy random greedy random"
    assert rows[1]["lineup"] == "random greedy random greedy"


def test_benchmark_reports_progress_after_each_round() -> None:
    updates: list[tuple[int, int]] = []

    run_benchmark(
        agent_a="greedy",
        agent_b="random",
        round_pairs=1,
        seed=2,
        dealer=0,
        config=_fast_config(),
        progress=lambda completed, total: updates.append((completed, total)),
    )

    assert updates == [(1, 2), (2, 2)]


def test_progress_bar_renders_to_stream() -> None:
    stream = io.StringIO()
    progress_bar = ProgressBar(4, stream=stream, width=10)

    progress_bar.update(1, 4)
    progress_bar.update(4, 4)
    progress_bar.close()

    output = stream.getvalue()
    assert "Benchmark progress" in output
    assert "1/4" in output
    assert "4/4" in output
    assert output.endswith("\n")
