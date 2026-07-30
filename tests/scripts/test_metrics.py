from __future__ import annotations

import csv
import json

from scripts.metrics import MetricsWriter


def test_metrics_writer_outputs_jsonl_and_csv(tmp_path) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    csv_path = tmp_path / "metrics.csv"

    with MetricsWriter(
        jsonl_path=str(jsonl_path),
        csv_path=str(csv_path),
        csv_fields=("event", "round", "value"),
    ) as writer:
        writer.write({"event": "train", "round": 1, "value": 0.25})
        writer.write({"event": "eval", "round": 2, "value": -1.5})

    jsonl_rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert jsonl_rows == [
        {"event": "train", "round": 1, "value": 0.25},
        {"event": "eval", "round": 2, "value": -1.5},
    ]

    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows == [
        {"event": "train", "round": "1", "value": "0.25"},
        {"event": "eval", "round": "2", "value": "-1.5"},
    ]
