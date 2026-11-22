"""Tests for the calibrated complex AWGN generator: equations (4.1), (4.2), (6.1).

Seed policy: every Monte Carlo test fixes its seed. Same seed, same samples.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfc

from wireless_phy_lab.channel import awgn
from wireless_phy_lab.modulation import qpsk

SEED = 20261123


def qfunc(x: np.ndarray | float) -> np.ndarray | float:
    """Gaussian tail probability, ``Q(x) = erfc(x / sqrt(2)) / 2`` (DLMF 7.2)."""
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2.0))


def test_noise_variance_is_inverse_es_n0() -> None:
    """sigma^2 = 1/(Es/N0), equation (6.1)."""
    es_n0 = np.array([0.5, 1.0, 10.0])
    np.testing.assert_allclose(awgn.noise_variance(es_n0), 1.0 / es_n0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(awgn.noise_variance(2.0) * 2.0, 1.0, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_noise_variance_rejects_invalid_es_n0(value: float) -> None:
    with pytest.raises(ValueError):
        awgn.noise_variance(value)


def test_noise_is_circularly_symmetric_at_the_requested_variance() -> None:
    """E|n|^2 = sigma^2, per-rail variance sigma^2/2, E[n^2] = 0, equation (4.2)."""
    rng = np.random.default_rng(SEED)
    es_n0 = 4.0
    sigma2 = float(awgn.noise_variance(es_n0))
    noise = awgn.generate_noise(1_000_000, es_n0, rng=rng)
    n_samples = noise.size
    real = noise.real
    imag = noise.imag

    mean_bound = 5.0 * np.sqrt(sigma2 / 2.0 / n_samples)
    assert abs(real.mean()) < mean_bound
    assert abs(imag.mean()) < mean_bound

    variance_bound = 5.0 * (sigma2 / 2.0) * np.sqrt(2.0 / (n_samples - 1))
    assert abs(real.var(ddof=1) - sigma2 / 2.0) < variance_bound
    assert abs(imag.var(ddof=1) - sigma2 / 2.0) < variance_bound

    energy_bound = 5.0 * sigma2 / np.sqrt(n_samples)
    assert abs(np.mean(np.abs(noise) ** 2) - sigma2) < energy_bound

    # Circularity: real and imaginary parts uncorrelated, and E[n^2] = 0.
    cross_bound = 5.0 * (sigma2 / 2.0) / np.sqrt(n_samples)
    assert abs(np.mean(real * imag)) < cross_bound
    pseudo_covariance_bound = 5.0 * np.sqrt(2.0) * sigma2 / np.sqrt(n_samples)
    assert abs(np.mean(noise**2)) < pseudo_covariance_bound


def test_generation_is_reproducible_for_a_fixed_seed() -> None:
    first = awgn.generate_noise((8, 3), 2.0, rng=np.random.default_rng(SEED))
    second = awgn.generate_noise((8, 3), 2.0, rng=np.random.default_rng(SEED))
    other = awgn.generate_noise((8, 3), 2.0, rng=np.random.default_rng(SEED + 1))

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_add_awgn_preserves_shape_and_does_not_mutate_the_signal() -> None:
    signal = np.full((3, 4), (1.0 + 1.0j) / np.sqrt(2.0), dtype=np.complex128)
    original = signal.copy()

    received = awgn.add_awgn(signal, 2.0, rng=np.random.default_rng(SEED))
    expected_noise = awgn.generate_noise(
        signal.shape, 2.0, rng=np.random.default_rng(SEED)
    )

    assert received.shape == signal.shape
    assert received.dtype == np.complex128
    np.testing.assert_array_equal(signal, original)
    np.testing.assert_allclose(received - signal, expected_noise, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("eb_n0_db", [0.0, 6.0])
def test_awgn_link_matches_closed_form(eb_n0_db: float) -> None:
    """Channel plus QPSK chain reproduces Q(sqrt(Es/N0)), equation (8.2)."""
    rng = np.random.default_rng(SEED + int(10.0 * eb_n0_db))
    n_bits = 1_000_000

    es_n0 = float(qpsk.es_n0_from_eb_n0(10.0 ** (eb_n0_db / 10.0)))
    bits = rng.integers(
        0, 2, size=(n_bits // qpsk.BITS_PER_SYMBOL, qpsk.BITS_PER_SYMBOL), dtype=np.uint8
    )
    received = awgn.add_awgn(qpsk.map_bits(bits), es_n0, rng=rng)
    measured = float(np.mean(qpsk.demap(received) != bits))
    expected = float(qfunc(np.sqrt(es_n0)))
    standard_error = np.sqrt(expected * (1.0 - expected) / n_bits)

    assert abs(measured - expected) <= 5.0 * standard_error, (
        f"Eb/N0={eb_n0_db} dB: measured {measured:.6e}, "
        f"closed form {expected:.6e}, standard error {standard_error:.2e}"
    )
