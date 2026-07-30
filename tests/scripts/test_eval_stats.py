from __future__ import annotations

import pytest

from scripts.eval_stats import mean_confidence_interval, wilson_score_interval


def test_mean_confidence_interval_contains_sample_mean() -> None:
    interval = mean_confidence_interval((1, 2, 3, 4), confidence=0.95)

    assert interval.low < 2.5 < interval.high
    assert interval.confidence == 0.95


def test_wilson_score_interval_contains_observed_rate() -> None:
    interval = wilson_score_interval(30, 100, confidence=0.95)

    assert interval.low < 0.3 < interval.high
    assert interval.confidence == 0.95


def test_confidence_intervals_reject_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        mean_confidence_interval((1, 2, 3), confidence=1.0)
