from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    confidence: float
    low: float
    high: float

    def to_dict(self) -> dict[str, float]:
        return {
            "confidence": self.confidence,
            "low": self.low,
            "high": self.high,
        }


def mean_confidence_interval(
    values: Sequence[int | float],
    *,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    _validate_confidence(confidence)
    if not values:
        return ConfidenceInterval(confidence, 0.0, 0.0)

    mean = statistics.fmean(values)
    if len(values) < 2:
        return ConfidenceInterval(confidence, mean, mean)

    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    radius = _z_value(confidence) * standard_error
    return ConfidenceInterval(confidence, mean - radius, mean + radius)


def wilson_score_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    _validate_confidence(confidence)
    if successes < 0:
        raise ValueError("successes cannot be negative")
    if total < 0:
        raise ValueError("total cannot be negative")
    if successes > total:
        raise ValueError("successes cannot exceed total")
    if total == 0:
        return ConfidenceInterval(confidence, 0.0, 0.0)

    z = _z_value(confidence)
    z2 = z * z
    proportion = successes / total
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * total)) / total)
        / denominator
    )
    return ConfidenceInterval(confidence, center - radius, center + radius)


def _validate_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")


def _z_value(confidence: float) -> float:
    return statistics.NormalDist().inv_cdf((1.0 + confidence) / 2.0)
