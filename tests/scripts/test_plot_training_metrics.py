from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plot_training_metrics import (
    best_evaluation,
    load_metric_run,
    load_metric_runs,
)


CSV_HEADER = (
    "event,round,train_mean_return,train_policy_loss,train_value_loss,"
    "train_entropy,eval_mean_score_diff,eval_mean_score_diff_ci_low,"
    "eval_mean_score_diff_ci_high,eval_agent_a_win_rate,"
    "eval_agent_a_win_rate_ci_low,eval_agent_a_win_rate_ci_high,"
    "eval_agent_a_double_victories,eval_agent_b_double_victories\n"
)


def test_load_metric_run_parses_csv_train_and_eval_rows(tmp_path: Path) -> None:
    path = tmp_path / "actor_critic.csv"
    path.write_text(
        CSV_HEADER
        + "train,10,-0.1,0.2,0.03,1.1,,,,,,,,\n"
        + "eval,20,,,,,5.5,-1.0,12.0,0.55,0.45,0.64,12,8\n",
        encoding="utf-8",
    )

    run = load_metric_run(path)

    assert run.label == tmp_path.name
    assert len(run.training) == 1
    assert run.training[0].round == 10
    assert run.training[0].mean_return == -0.1
    assert len(run.evaluations) == 1
    assert run.evaluations[0].round == 20
    assert run.evaluations[0].mean_score_diff == 5.5
    assert run.evaluations[0].agent_a_double_victories == 12


def test_load_metric_run_parses_jsonl_and_selects_best_eval(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    records = [
        {
            "event": "eval",
            "round": 100,
            "eval_mean_score_diff": -10.0,
            "eval_agent_a_win_rate": 0.4,
        },
        {
            "event": "eval",
            "round": 200,
            "eval_mean_score_diff": 12.0,
            "eval_agent_a_win_rate": 0.6,
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    run = load_metric_run(path, label="run-a")
    best = best_evaluation(run)

    assert run.label == "run-a"
    assert best is not None
    assert best.round == 200
    assert best.mean_score_diff == 12.0


def test_load_metric_runs_requires_one_label_per_file(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(CSV_HEADER, encoding="utf-8")
    second.write_text(CSV_HEADER, encoding="utf-8")

    with pytest.raises(ValueError, match="one label per metrics file"):
        load_metric_runs((first, second), labels=("only-one",))
