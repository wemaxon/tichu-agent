from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ImageFormat = Literal["png", "svg"]
PlotName = Literal[
    "eval_score",
    "eval_win_rate",
    "training_diagnostics",
    "double_victories",
]


@dataclass(frozen=True, slots=True)
class TrainingPoint:
    round: int
    mean_return: float | None
    policy_loss: float | None
    value_loss: float | None
    entropy: float | None


@dataclass(frozen=True, slots=True)
class EvaluationPoint:
    round: int
    mean_score_diff: float
    mean_score_diff_ci_low: float | None
    mean_score_diff_ci_high: float | None
    win_rate: float
    win_rate_ci_low: float | None
    win_rate_ci_high: float | None
    agent_a_double_victories: int | None
    agent_b_double_victories: int | None


@dataclass(frozen=True, slots=True)
class MetricRun:
    label: str
    path: Path
    training: tuple[TrainingPoint, ...]
    evaluations: tuple[EvaluationPoint, ...]


class AxesLike(Protocol):
    transAxes: Any

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def fill_between(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def axhline(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def scatter(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def text(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def set_title(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def set_xlabel(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def set_ylabel(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def grid(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def legend(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def get_legend_handles_labels(self, *args: Any, **kwargs: Any) -> Any:
        ...


def load_metric_runs(
    paths: tuple[Path, ...],
    *,
    labels: tuple[str, ...] | None = None,
) -> tuple[MetricRun, ...]:
    if labels is not None and len(labels) != len(paths):
        raise ValueError("--labels must contain exactly one label per metrics file")

    runs: list[MetricRun] = []
    for index, path in enumerate(paths):
        label = labels[index] if labels is not None else _default_label(path)
        runs.append(load_metric_run(path, label=label))
    return tuple(runs)


def load_metric_run(path: Path, *, label: str | None = None) -> MetricRun:
    rows = _read_metric_rows(path)
    training: list[TrainingPoint] = []
    evaluations: list[EvaluationPoint] = []

    for row in rows:
        event = str(row.get("event", ""))
        if event == "train":
            training.append(
                TrainingPoint(
                    round=_required_int(row, "round"),
                    mean_return=_optional_float(row, "train_mean_return"),
                    policy_loss=_optional_float(row, "train_policy_loss"),
                    value_loss=_optional_float(row, "train_value_loss"),
                    entropy=_optional_float(row, "train_entropy"),
                )
            )
        elif event == "eval":
            evaluations.append(
                EvaluationPoint(
                    round=_required_int(row, "round"),
                    mean_score_diff=_required_float(row, "eval_mean_score_diff"),
                    mean_score_diff_ci_low=_optional_float(
                        row,
                        "eval_mean_score_diff_ci_low",
                    ),
                    mean_score_diff_ci_high=_optional_float(
                        row,
                        "eval_mean_score_diff_ci_high",
                    ),
                    win_rate=_required_float(row, "eval_agent_a_win_rate"),
                    win_rate_ci_low=_optional_float(row, "eval_agent_a_win_rate_ci_low"),
                    win_rate_ci_high=_optional_float(row, "eval_agent_a_win_rate_ci_high"),
                    agent_a_double_victories=_optional_int(
                        row,
                        "eval_agent_a_double_victories",
                    ),
                    agent_b_double_victories=_optional_int(
                        row,
                        "eval_agent_b_double_victories",
                    ),
                )
            )

    return MetricRun(
        label=_default_label(path) if label is None else label,
        path=path,
        training=tuple(training),
        evaluations=tuple(evaluations),
    )


def best_evaluation(run: MetricRun) -> EvaluationPoint | None:
    if not run.evaluations:
        return None
    return max(run.evaluations, key=lambda point: point.mean_score_diff)


def plot_training_metrics(
    runs: tuple[MetricRun, ...],
    *,
    output_dir: Path,
    image_format: ImageFormat = "png",
    dpi: int = 150,
) -> tuple[Path, ...]:
    if not runs:
        raise ValueError("at least one metrics run is required")
    if dpi < 1:
        raise ValueError("dpi must be positive")

    pyplot = _load_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = (
        _plot_eval_score(pyplot, runs, output_dir, image_format, dpi),
        _plot_eval_win_rate(pyplot, runs, output_dir, image_format, dpi),
        _plot_training_diagnostics(pyplot, runs, output_dir, image_format, dpi),
        _plot_double_victories(pyplot, runs, output_dir, image_format, dpi),
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot actor-critic training metrics from CSV or JSONL logs"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="One or more metrics CSV/JSONL files",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional labels, one per metrics file",
    )
    parser.add_argument("--out", required=True, help="Output directory for plots")
    parser.add_argument("--format", choices=("png", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    paths = tuple(Path(path) for path in args.metrics)
    labels = None if args.labels is None else tuple(str(label) for label in args.labels)
    try:
        runs = load_metric_runs(paths, labels=labels)
        outputs = plot_training_metrics(
            runs,
            output_dir=Path(args.out),
            image_format=args.format,
            dpi=args.dpi,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2

    for path in outputs:
        print(path)
    return 0


def _read_metric_rows(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        raise ValueError(f"metrics file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))
    if suffix == ".jsonl":
        rows: list[dict[str, object]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} in {path} is not a JSON object")
                rows.append(dict(value))
        return tuple(rows)
    raise ValueError(f"unsupported metrics file type: {path}")


def _plot_eval_score(
    pyplot: Any,
    runs: tuple[MetricRun, ...],
    output_dir: Path,
    image_format: ImageFormat,
    dpi: int,
) -> Path:
    fig, ax = pyplot.subplots(figsize=(9, 5))
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    for run in runs:
        points = run.evaluations
        if not points:
            continue
        rounds = [point.round for point in points]
        values = [point.mean_score_diff for point in points]
        (line,) = ax.plot(rounds, values, marker="o", linewidth=1.5, label=run.label)
        lows = _all_floats(point.mean_score_diff_ci_low for point in points)
        highs = _all_floats(point.mean_score_diff_ci_high for point in points)
        if lows is not None and highs is not None:
            ax.fill_between(
                rounds,
                lows,
                highs,
                color=line.get_color(),
                alpha=0.15,
                linewidth=0,
            )
        best = best_evaluation(run)
        if best is not None:
            ax.scatter(
                [best.round],
                [best.mean_score_diff],
                color=line.get_color(),
                edgecolor="black",
                zorder=5,
            )
    _finish_axis(ax, "Evaluation Score", "train round", "mean score diff / round")
    _ensure_nonempty(ax, runs, has_data=any(run.evaluations for run in runs))
    return _save_figure(fig, output_dir, "eval_score", image_format, dpi)


def _plot_eval_win_rate(
    pyplot: Any,
    runs: tuple[MetricRun, ...],
    output_dir: Path,
    image_format: ImageFormat,
    dpi: int,
) -> Path:
    fig, ax = pyplot.subplots(figsize=(9, 5))
    ax.axhline(0.5, color="black", linewidth=1, alpha=0.5)
    for run in runs:
        points = run.evaluations
        if not points:
            continue
        rounds = [point.round for point in points]
        values = [point.win_rate for point in points]
        (line,) = ax.plot(rounds, values, marker="o", linewidth=1.5, label=run.label)
        lows = _all_floats(point.win_rate_ci_low for point in points)
        highs = _all_floats(point.win_rate_ci_high for point in points)
        if lows is not None and highs is not None:
            ax.fill_between(
                rounds,
                lows,
                highs,
                color=line.get_color(),
                alpha=0.15,
                linewidth=0,
            )
    ax.set_ylim(-0.05, 1.05)
    _finish_axis(ax, "Evaluation Win Rate", "train round", "A win rate")
    _ensure_nonempty(ax, runs, has_data=any(run.evaluations for run in runs))
    return _save_figure(fig, output_dir, "eval_win_rate", image_format, dpi)


def _plot_training_diagnostics(
    pyplot: Any,
    runs: tuple[MetricRun, ...],
    output_dir: Path,
    image_format: ImageFormat,
    dpi: int,
) -> Path:
    fig, axes = pyplot.subplots(2, 2, figsize=(11, 7), sharex=True)
    diagnostics = (
        ("mean_return", "Train Mean Return", "return"),
        ("policy_loss", "Policy Loss", "loss"),
        ("value_loss", "Value Loss", "loss"),
        ("entropy", "Entropy", "entropy"),
    )
    for ax, (field, title, ylabel) in zip(axes.flat, diagnostics, strict=True):
        _plot_training_field(ax, runs, field, title, ylabel)
    fig.suptitle("Training Diagnostics")
    fig.tight_layout()
    return _save_figure(fig, output_dir, "training_diagnostics", image_format, dpi)


def _plot_double_victories(
    pyplot: Any,
    runs: tuple[MetricRun, ...],
    output_dir: Path,
    image_format: ImageFormat,
    dpi: int,
) -> Path:
    fig, ax = pyplot.subplots(figsize=(9, 5))
    has_data = False
    for run in runs:
        rounds: list[int] = []
        agent_a_counts: list[int] = []
        agent_b_counts: list[int] = []
        for point in run.evaluations:
            a_count = point.agent_a_double_victories
            b_count = point.agent_b_double_victories
            if a_count is None or b_count is None:
                continue
            rounds.append(point.round)
            agent_a_counts.append(a_count)
            agent_b_counts.append(b_count)
        if not rounds:
            continue
        has_data = True
        (line,) = ax.plot(
            rounds,
            agent_a_counts,
            marker="o",
            linewidth=1.5,
            label=f"{run.label} A",
        )
        ax.plot(
            rounds,
            agent_b_counts,
            marker="x",
            linestyle="--",
            linewidth=1.2,
            color=line.get_color(),
            label=f"{run.label} B",
        )
    _finish_axis(ax, "Double Victories", "train round", "count in evaluation")
    _ensure_nonempty(ax, runs, has_data=has_data)
    return _save_figure(fig, output_dir, "double_victories", image_format, dpi)


def _plot_training_field(
    ax: AxesLike,
    runs: tuple[MetricRun, ...],
    field: str,
    title: str,
    ylabel: str,
) -> None:
    has_data = False
    for run in runs:
        points = [
            (point.round, getattr(point, field))
            for point in run.training
            if getattr(point, field) is not None
        ]
        if not points:
            continue
        has_data = True
        ax.plot(
            [point[0] for point in points],
            [float(point[1]) for point in points],
            linewidth=1.2,
            label=run.label,
        )
    _finish_axis(ax, title, "train round", ylabel)
    _ensure_nonempty(ax, runs, has_data=has_data)


def _finish_axis(ax: AxesLike, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def _ensure_nonempty(
    ax: AxesLike,
    runs: tuple[MetricRun, ...],
    *,
    has_data: bool,
) -> None:
    if has_data:
        return
    labels = ", ".join(run.label for run in runs)
    ax.text(
        0.5,
        0.5,
        f"No matching records: {labels}",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )


def _save_figure(
    fig: Any,
    output_dir: Path,
    name: PlotName,
    image_format: ImageFormat,
    dpi: int,
) -> Path:
    path = output_dir / f"{name}.{image_format}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.clf()
    return path


def _load_pyplot() -> Any:
    try:
        config_dir = Path(tempfile.gettempdir()) / "tichu-ai-matplotlib"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(config_dir))

        import matplotlib  # type: ignore[import-not-found]

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting; install requirements.txt"
        ) from exc
    return pyplot


def _all_floats(values: Any) -> list[float] | None:
    result: list[float] = []
    for value in values:
        if value is None:
            return None
        result.append(value)
    return result


def _default_label(path: Path) -> str:
    if path.name in {"actor_critic.csv", "actor_critic.jsonl"} and path.parent.name:
        return path.parent.name
    return path.stem


def _required_int(row: dict[str, object], key: str) -> int:
    value = _required_float(row, key)
    if not value.is_integer():
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _optional_int(row: dict[str, object], key: str) -> int | None:
    value = _optional_float(row, key)
    if value is None:
        return None
    if not value.is_integer():
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _required_float(row: dict[str, object], key: str) -> float:
    value = _optional_float(row, key)
    if value is None:
        raise ValueError(f"missing required metric field: {key}")
    return value


def _optional_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    if isinstance(value, int | float | str):
        return float(value)
    raise ValueError(f"{key} must be numeric")


if __name__ == "__main__":
    raise SystemExit(main())
