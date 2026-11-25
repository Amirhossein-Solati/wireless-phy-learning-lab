"""Tests for independent-replicate BER summaries."""

from __future__ import annotations

import numpy as np
import pytest

from wireless_phy_lab.detection import zf_detect
from wireless_phy_lab.evaluation.ber import ber_estimate
from wireless_phy_lab.evaluation.replicates import (
    simulate_ber_replicates,
    summarize_replicates,
    t_interval,
)

SEED = 20261128


def test_t_interval_contains_mean_and_is_symmetric() -> None:
    values = np.array([0.010, 0.012, 0.011, 0.013, 0.009])
    mean, low, high = t_interval(values, confidence=0.95)
    assert mean == pytest.approx(float(values.mean()))
    assert low < mean < high
    assert (mean - low) == pytest.approx(high - mean, rel=1e-12)


def test_t_interval_zero_variance_collapses() -> None:
    mean, low, high = t_interval([0.1, 0.1, 0.1], confidence=0.95)
    assert mean == pytest.approx(0.1)
    assert low == pytest.approx(0.1)
    assert high == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("values", "confidence"),
    [
        ([0.1], 0.95),
        ([], 0.95),
        ([-0.1, 0.2], 0.95),
        ([1.1, 0.2], 0.95),
        ([0.1, 0.2], 0.0),
        ([0.1, 0.2], 1.0),
    ],
)
def test_t_interval_rejects_invalid_input(values: list[float], confidence: float) -> None:
    with pytest.raises(ValueError):
        t_interval(values, confidence=confidence)


def test_summarize_replicates_uses_mean_and_pooled_counts() -> None:
    estimates = [ber_estimate(10, 1000), ber_estimate(20, 1000), ber_estimate(30, 1000)]
    summary = summarize_replicates(estimates, confidence=0.95)

    assert summary.replicates == 3
    assert summary.ber == pytest.approx(0.02)
    assert summary.total_errors == 60
    assert summary.total_bits == 3000
    assert summary.ci_low <= summary.ber <= summary.ci_high
    assert len(summary.replicate_bers) == 3


def _zf(channel: np.ndarray, received: np.ndarray, noise_variance: float) -> np.ndarray:
    return zf_detect(channel, received)


def test_replicate_simulation_is_reproducible() -> None:
    kwargs = {
        "nr": 2,
        "nt": 2,
        "eb_n0_db": 6.0,
        "seed": SEED,
        "replicates": 5,
        "target_errors": 10,
        "max_bits": 5000,
        "batch_symbols": 500,
        "confidence": 0.95,
    }
    first = simulate_ber_replicates(_zf, **kwargs)
    second = simulate_ber_replicates(_zf, **kwargs)
    other = simulate_ber_replicates(_zf, **{**kwargs, "seed": SEED + 1})

    assert first == second
    assert first.replicate_seeds != other.replicate_seeds
    assert first.aggregate.ber != other.aggregate.ber


def test_replicate_summary_accounting() -> None:
    point = simulate_ber_replicates(
        _zf,
        nr=2,
        nt=2,
        eb_n0_db=6.0,
        seed=SEED,
        replicates=5,
        target_errors=10,
        max_bits=5000,
        batch_symbols=500,
    )

    assert len(point.replicate_seeds) == 5
    assert len(set(point.replicate_seeds)) == 5
    assert len(point.per_stream) == 2
    assert point.aggregate.ci_low <= point.aggregate.ber <= point.aggregate.ci_high
    assert sum(item.total_errors for item in point.per_stream) == point.aggregate.total_errors
    assert sum(item.total_bits for item in point.per_stream) == point.aggregate.total_bits
    assert all(
        item.ci_low <= item.ber <= item.ci_high for item in point.per_stream
    )


def test_replicates_must_be_at_least_two() -> None:
    with pytest.raises(ValueError):
        simulate_ber_replicates(
            _zf,
            nr=2,
            nt=2,
            eb_n0_db=6.0,
            seed=SEED,
            replicates=1,
        )
