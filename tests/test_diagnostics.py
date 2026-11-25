"""Tests for channel condition-number diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from wireless_phy_lab.evaluation.diagnostics import condition_number_summary


def test_siso_condition_number_is_one() -> None:
    summary = condition_number_summary(1, 1, seed=7, samples=100)
    assert summary.median == pytest.approx(1.0)
    assert summary.p95 == pytest.approx(1.0)
    assert summary.maximum == pytest.approx(1.0)
    assert summary.fraction_above == 0.0
    assert summary.samples == 100
    assert summary.seed == 7


def test_summary_is_reproducible() -> None:
    first = condition_number_summary(4, 4, seed=20261128, samples=1000)
    second = condition_number_summary(4, 4, seed=20261128, samples=1000)
    other = condition_number_summary(4, 4, seed=20261129, samples=1000)

    assert first == second
    assert first != other


def test_summary_fields_are_ordered() -> None:
    summary = condition_number_summary(4, 4, seed=20261128, samples=5000, threshold=100.0)
    assert 1.0 <= summary.median <= summary.p95 <= summary.maximum
    assert 0.0 <= summary.fraction_above <= 1.0
    assert summary.threshold == 100.0


@pytest.mark.parametrize("samples", [0, -1, 1.5, True])
def test_invalid_sample_count_is_rejected(samples: object) -> None:
    with pytest.raises(ValueError):
        condition_number_summary(2, 2, seed=1, samples=samples)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [1.0, 0.0, -5.0, np.inf, np.nan])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        condition_number_summary(2, 2, seed=1, samples=10, threshold=threshold)
