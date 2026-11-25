"""Tests for exhaustive QPSK maximum-likelihood detection."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from wireless_phy_lab.detection import ml_detect, mmse_detect, zf_detect
from wireless_phy_lab.detection.ml import _qpsk_candidates
from wireless_phy_lab.modulation import qpsk

SEED = 20261129


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (rng.standard_normal(shape) + 1.0j * rng.standard_normal(shape)) / np.sqrt(2.0)


@pytest.mark.parametrize(("nr", "nt"), [(2, 2), (4, 2), (4, 4)])
def test_ml_recovers_qpsk_without_noise(nr: int, nt: int) -> None:
    rng = np.random.default_rng(SEED)
    channel = _complex_normal(rng, (nr, nt))
    bits = rng.integers(0, 2, size=(nt, qpsk.BITS_PER_SYMBOL), dtype=np.uint8)
    symbols = qpsk.map_bits(bits)

    estimates = ml_detect(channel, channel @ symbols)
    np.testing.assert_array_equal(estimates, symbols)
    np.testing.assert_array_equal(qpsk.demap(estimates), bits)


def _brute_force_ml(channel: np.ndarray, received: np.ndarray) -> np.ndarray:
    nt = channel.shape[-1]
    candidates = list(itertools.product(qpsk.CONSTELLATION, repeat=nt))
    metrics = [
        np.sum(np.abs(received - channel @ np.asarray(candidate)) ** 2)
        for candidate in candidates
    ]
    return np.asarray(candidates[int(np.argmin(metrics))])


def test_ml_matches_brute_force_definition() -> None:
    rng = np.random.default_rng(SEED + 1)
    for _ in range(5):
        channel = _complex_normal(rng, (4, 4))
        received = _complex_normal(rng, (4,))
        np.testing.assert_array_equal(
            ml_detect(channel, received), _brute_force_ml(channel, received)
        )


def test_ml_metric_is_no_worse_than_linear_hard_decisions() -> None:
    rng = np.random.default_rng(SEED + 2)
    channel = _complex_normal(rng, (4, 4))
    received = _complex_normal(rng, (4,))

    ml_symbols = ml_detect(channel, received)
    zf_symbols = qpsk.map_bits(qpsk.demap(zf_detect(channel, received)))
    mmse_symbols = qpsk.map_bits(qpsk.demap(mmse_detect(channel, received, 0.5)))

    ml_metric = np.sum(np.abs(received - channel @ ml_symbols) ** 2)
    zf_metric = np.sum(np.abs(received - channel @ zf_symbols) ** 2)
    mmse_metric = np.sum(np.abs(received - channel @ mmse_symbols) ** 2)

    assert ml_metric <= zf_metric
    assert ml_metric <= mmse_metric


def test_batched_ml_equals_per_channel_loop() -> None:
    rng = np.random.default_rng(SEED + 3)
    channels = _complex_normal(rng, (6, 2, 2))
    received = _complex_normal(rng, (6, 2))

    actual = ml_detect(channels, received)
    expected = np.stack([ml_detect(channels[k], received[k]) for k in range(6)])
    np.testing.assert_array_equal(actual, expected)


def test_candidate_set_has_four_to_the_nt_vectors() -> None:
    for nt in (1, 2, 4):
        candidates = _qpsk_candidates(nt)
        assert candidates.shape == (4**nt, nt)
        assert len({tuple(candidate) for candidate in candidates}) == 4**nt
        assert np.all(np.isin(candidates, qpsk.CONSTELLATION))


@pytest.mark.parametrize(
    ("channel", "received"),
    [
        (np.zeros(2), np.zeros(2)),
        (np.zeros((2, 2)), np.zeros((2, 1))),
        (np.zeros((3, 2, 2)), np.zeros(2)),
        (np.zeros((3, 2)), np.zeros(2)),
    ],
)
def test_ml_rejects_inconsistent_shapes(
    channel: np.ndarray, received: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        ml_detect(channel, received)
