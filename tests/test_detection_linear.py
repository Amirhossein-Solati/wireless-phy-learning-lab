"""Tests for zero forcing and biased MMSE, equations (7.1)-(7.7).

The reference in every comparison test is the explicit matrix definition

    ZF:    G = (H^H H)^{-1} H^H
    MMSE:  G = (H^H H + sigma^2 I)^{-1} H^H

computed with ``numpy.linalg.inv`` on small, well-conditioned matrices. The
implementation under test solves the normal equations directly and never forms
that inverse.

Seed policy: every Monte Carlo test fixes its seed. Statistical assertions use
five standard errors of the estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from wireless_phy_lab.detection import condition_number, mmse_detect, zf_detect
from wireless_phy_lab.modulation import qpsk

SEED = 20261126


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (rng.standard_normal(shape) + 1.0j * rng.standard_normal(shape)) / np.sqrt(2.0)


def _random_channel(rng: np.random.Generator, nr: int, nt: int) -> np.ndarray:
    return _complex_normal(rng, (nr, nt))


def _unitary_channel(n: int) -> np.ndarray:
    return np.fft.fft(np.eye(n)) / np.sqrt(n)


@pytest.mark.parametrize(("nr", "nt"), [(2, 2), (4, 2), (4, 4)])
def test_zf_matches_explicit_definition(nr: int, nt: int) -> None:
    rng = np.random.default_rng(SEED)
    channel = _random_channel(rng, nr, nt)
    assert np.linalg.cond(channel) < 20.0
    received = _complex_normal(rng, (nr,))

    gram = channel.conj().T @ channel
    filter_matrix = np.linalg.inv(gram) @ channel.conj().T
    expected = filter_matrix @ received

    actual = zf_detect(channel, received)
    assert actual.shape == (nt,)
    assert actual.dtype == np.complex128
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        actual, np.linalg.pinv(channel) @ received, rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize(("nr", "nt"), [(2, 2), (4, 4)])
@pytest.mark.parametrize("noise_variance", [0.0, 0.1, 1.0, 10.0])
def test_mmse_matches_explicit_definition(
    nr: int, nt: int, noise_variance: float
) -> None:
    rng = np.random.default_rng(SEED + 1)
    channel = _random_channel(rng, nr, nt)
    assert np.linalg.cond(channel) < 20.0
    received = _complex_normal(rng, (nr,))

    gram = channel.conj().T @ channel
    filter_matrix = (
        np.linalg.inv(gram + noise_variance * np.eye(nt)) @ channel.conj().T
    )
    expected = filter_matrix @ received

    actual = mmse_detect(channel, received, noise_variance)
    assert actual.shape == (nt,)
    assert actual.dtype == np.complex128
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_batched_detection_equals_per_channel_loop() -> None:
    rng = np.random.default_rng(SEED + 2)
    channels = _complex_normal(rng, (6, 2, 2))
    received = _complex_normal(rng, (6, 2))

    expected_zf = np.stack([zf_detect(channels[k], received[k]) for k in range(6)])
    actual_zf = zf_detect(channels, received)
    assert actual_zf.shape == (6, 2)
    np.testing.assert_allclose(actual_zf, expected_zf, rtol=1e-12, atol=1e-12)

    sigma2 = 0.5
    expected_mmse = np.stack(
        [mmse_detect(channels[k], received[k], sigma2) for k in range(6)]
    )
    actual_mmse = mmse_detect(channels, received, sigma2)
    assert actual_mmse.shape == (6, 2)
    np.testing.assert_allclose(actual_mmse, expected_mmse, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("nr", "nt"), [(2, 2), (4, 4)])
def test_zf_recovers_qpsk_without_noise(nr: int, nt: int) -> None:
    rng = np.random.default_rng(SEED + 3)
    channel = _random_channel(rng, nr, nt)
    assert np.linalg.cond(channel) < 20.0
    bits = rng.integers(0, 2, size=(nt, qpsk.BITS_PER_SYMBOL), dtype=np.uint8)
    symbols = qpsk.map_bits(bits)

    estimates = zf_detect(channel, channel @ symbols)
    np.testing.assert_allclose(estimates, symbols, rtol=0.0, atol=1e-11)
    np.testing.assert_array_equal(qpsk.demap(estimates), bits)


def test_zf_identity_channel_reduces_to_awgn() -> None:
    rng = np.random.default_rng(SEED + 4)
    received = _complex_normal(rng, (4,))
    estimates = zf_detect(np.eye(4, dtype=np.complex128), received)

    np.testing.assert_allclose(estimates, received, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(qpsk.demap(estimates), qpsk.demap(received))


def test_mmse_identity_channel_scales_awgn_decisions() -> None:
    rng = np.random.default_rng(SEED + 5)
    received = _complex_normal(rng, (4,))
    sigma2 = 0.5
    estimates = mmse_detect(np.eye(4, dtype=np.complex128), received, sigma2)

    np.testing.assert_allclose(estimates, received / (1.0 + sigma2), rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(qpsk.demap(estimates), qpsk.demap(received))


@pytest.mark.parametrize("n", [2, 4])
def test_mmse_converges_to_zf_at_high_snr(n: int) -> None:
    rng = np.random.default_rng(SEED + 6)
    channel = _unitary_channel(n)
    received = _complex_normal(rng, (n,))
    sigma2 = 1e-6

    np.testing.assert_allclose(
        mmse_detect(channel, received, sigma2),
        zf_detect(channel, received),
        rtol=1.1e-6,
        atol=0.0,
    )


@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("noise_variance", [0.1, 1.0, 10.0])
def test_mmse_bias_is_real_positive_and_unit_bounded(
    n: int, noise_variance: float
) -> None:
    rng = np.random.default_rng(SEED + 7)
    channel = _random_channel(rng, n, n)
    assert np.linalg.cond(channel) < 20.0
    received = _complex_normal(rng, (n,))

    gram = channel.conj().T @ channel
    regularised = gram + noise_variance * np.eye(n)
    filter_matrix = np.linalg.inv(regularised) @ channel.conj().T
    gain = filter_matrix @ channel
    bias = np.diag(gain)

    assert np.max(np.abs(bias.imag)) < 1e-12
    assert np.all(bias.real > 0.0)
    assert np.all(bias.real <= 1.0)

    expected_bias = 1.0 - noise_variance * np.real(np.diag(np.linalg.inv(regularised)))
    np.testing.assert_allclose(bias.real, expected_bias, rtol=1e-10, atol=1e-12)

    biased_estimates = filter_matrix @ received
    np.testing.assert_array_equal(
        qpsk.demap(biased_estimates), qpsk.demap(biased_estimates / bias)
    )


def test_zf_error_covariance_matches_definition() -> None:
    """E[(x_hat - x)(x_hat - x)^H] = sigma^2 (H^H H)^{-1}, equation (7.2)."""
    rng = np.random.default_rng(SEED + 8)
    channel = _random_channel(rng, 2, 2)
    assert np.linalg.cond(channel) < 20.0
    bits = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    symbols = qpsk.map_bits(bits)

    sigma2 = 0.5
    n_samples = 100_000
    noise = np.sqrt(sigma2 / 2.0) * (
        rng.standard_normal((n_samples, 2)) + 1.0j * rng.standard_normal((n_samples, 2))
    )
    received = channel @ symbols + noise

    channels = np.broadcast_to(channel, (n_samples, 2, 2))
    errors = zf_detect(channels, received) - symbols

    empirical = errors.T @ errors.conj() / n_samples
    theoretical = sigma2 * np.linalg.inv(channel.conj().T @ channel)
    diagonal = np.real(np.diag(theoretical))
    bound = 5.0 * np.sqrt(np.outer(diagonal, diagonal) / n_samples)

    assert np.all(np.abs(empirical - theoretical) <= bound)


def test_mmse_regularizes_rank_deficient_channel_while_zf_raises() -> None:
    channel = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)
    received = np.array([2.0, 2.0], dtype=np.complex128)

    with pytest.raises(np.linalg.LinAlgError):
        zf_detect(channel, received)

    estimates = mmse_detect(channel, received, 0.1)
    assert estimates.shape == (2,)
    assert np.all(np.isfinite(estimates))


def test_zf_rejects_underdetermined_channel() -> None:
    channel = np.ones((2, 4), dtype=np.complex128)
    with pytest.raises(ValueError, match="Nr >= Nt"):
        zf_detect(channel, np.ones(2, dtype=np.complex128))


def test_mmse_handles_underdetermined_channel_with_regularization() -> None:
    rng = np.random.default_rng(SEED + 9)
    channel = _random_channel(rng, 2, 4)
    received = _complex_normal(rng, (2,))
    estimates = mmse_detect(channel, received, 0.5)

    assert estimates.shape == (4,)
    assert np.all(np.isfinite(estimates))


def test_mmse_with_zero_variance_equals_zf() -> None:
    rng = np.random.default_rng(SEED + 10)
    channel = _random_channel(rng, 4, 4)
    assert np.linalg.cond(channel) < 20.0
    received = _complex_normal(rng, (4,))

    np.testing.assert_allclose(
        mmse_detect(channel, received, 0.0),
        zf_detect(channel, received),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("channel", "received"),
    [
        (np.zeros(2), np.zeros(2)),
        (np.zeros((2, 2)), np.zeros((2, 1))),
        (np.zeros((3, 2, 2)), np.zeros(2)),
        (np.zeros((3, 2)), np.zeros(2)),
        (np.zeros((2, 2)), np.zeros(3)),
    ],
)
def test_detection_rejects_inconsistent_shapes(
    channel: np.ndarray, received: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        zf_detect(channel, received)
    with pytest.raises(ValueError):
        mmse_detect(channel, received, 0.5)


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_mmse_rejects_invalid_noise_variance(value: float) -> None:
    channel = np.eye(2, dtype=np.complex128)
    received = np.ones(2, dtype=np.complex128)
    with pytest.raises(ValueError):
        mmse_detect(channel, received, value)


def test_condition_number_matches_definition() -> None:
    np.testing.assert_allclose(condition_number(np.eye(3)), 1.0)
    np.testing.assert_allclose(condition_number(np.diag([1.0, 3.0])), 3.0)

    batch = np.stack([np.eye(2), np.diag([1.0, 4.0])])
    np.testing.assert_allclose(condition_number(batch), [1.0, 4.0])


def test_condition_number_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        condition_number(np.zeros(3))
    with pytest.raises(ValueError):
        condition_number(np.zeros((2, 0)))
