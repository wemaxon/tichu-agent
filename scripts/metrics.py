from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO, TypeAlias


MetricValue: TypeAlias = str | int | float | bool | None
MetricRecord: TypeAlias = Mapping[str, MetricValue]


class MetricsWriter:
    def __init__(
        self,
        *,
        jsonl_path: str | None = None,
        csv_path: str | None = None,
        csv_fields: Sequence[str] = (),
    ) -> None:
        self._jsonl_handle: TextIO | None = None
        self._csv_handle: TextIO | None = None
        self._csv_writer: csv.DictWriter[str] | None = None

        if jsonl_path is not None:
            path = Path(jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_handle = path.open("w", encoding="utf-8")

        if csv_path is not None:
            if not csv_fields:
                raise ValueError("csv_fields are required when csv_path is provided")
            path = Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_handle = path.open("w", encoding="utf-8", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_handle,
                fieldnames=tuple(csv_fields),
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()

    def write(self, record: MetricRecord) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
            self._jsonl_handle.flush()
        if self._csv_writer is not None:
            self._csv_writer.writerow(dict(record))
            assert self._csv_handle is not None
            self._csv_handle.flush()

    def close(self) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
            self._jsonl_handle = None
        if self._csv_handle is not None:
            self._csv_handle.close()
            self._csv_handle = None

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
