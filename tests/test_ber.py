"""Tests for bit-error counting and Wilson intervals."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binomtest

from wireless_phy_lab.evaluation.ber import (
    ber_estimate,
    count_bit_errors,
    wilson_interval,
)


def test_count_bit_errors_known() -> None:
    transmitted = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
    estimated = np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8)
    assert count_bit_errors(transmitted, estimated) == 3


def test_count_bit_errors_accepts_bool_arrays() -> None:
    transmitted = np.array([True, False, True])
    estimated = np.array([True, True, False])
    assert count_bit_errors(transmitted, estimated) == 2


@pytest.mark.parametrize(
    ("transmitted", "estimated"),
    [
        (np.zeros(2), np.zeros(3)),
        (np.zeros(0), np.zeros(0)),
        (np.array([0, 2]), np.array([0, 1])),
        (np.array([0, 1]), np.array([0, 2])),
    ],
)
def test_count_bit_errors_rejects_invalid_input(
    transmitted: np.ndarray, estimated: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        count_bit_errors(transmitted, estimated)


@pytest.mark.parametrize(
    ("errors", "bits"),
    [(0, 100), (1, 100), (50, 100), (99, 100), (100, 100), (10, 1000), (2000, 1_000_000)],
)
def test_wilson_interval_matches_scipy(errors: int, bits: int) -> None:
    lower, upper = wilson_interval(errors, bits, confidence=0.95)
    reference = binomtest(errors, bits).proportion_ci(
        confidence_level=0.95, method="wilson"
    )
    np.testing.assert_allclose([lower, upper], [reference.low, reference.high], rtol=1e-12)


def test_wilson_interval_is_bounded_and_contains_estimate() -> None:
    lower, upper = wilson_interval(37, 1000, confidence=0.99)
    assert 0.0 <= lower < 37 / 1000 < upper <= 1.0


def test_ber_estimate_fields_and_dictionary() -> None:
    estimate = ber_estimate(10, 1000, confidence=0.95)
    assert estimate.errors == 10
    assert estimate.bits == 1000
    assert estimate.ber == pytest.approx(0.01)
    assert estimate.ci_low < estimate.ber < estimate.ci_high
    assert estimate.confidence == 0.95

    payload = estimate.as_dict()
    assert payload["errors"] == 10
    assert payload["bits"] == 1000
    assert payload["ber"] == pytest.approx(0.01)
    assert set(payload) == {"errors", "bits", "ber", "ci_low", "ci_high", "confidence"}


@pytest.mark.parametrize(("errors", "bits"), [(-1, 100), (101, 100), (0, 0), (1, -1)])
def test_invalid_counts_are_rejected(errors: int, bits: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(errors, bits)
    with pytest.raises(ValueError):
        ber_estimate(errors, bits)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf])
def test_invalid_confidence_is_rejected(confidence: float) -> None:
    with pytest.raises(ValueError):
        wilson_interval(10, 100, confidence=confidence)
