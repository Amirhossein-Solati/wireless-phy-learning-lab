"""Tests for the flat Rayleigh channel: equations (3.1)-(3.3), decision D-02.

Seed policy: every Monte Carlo test fixes its seed. Same seed, same samples.
Statistical assertions use five standard errors of the estimator; for a fixed
seed they are deterministic, not flaky.
"""

from __future__ import annotations

import numpy as np
import pytest

from wireless_phy_lab.channel import rayleigh

SEED = 20261124


def test_shape_single_and_batched() -> None:
    assert rayleigh.generate_rayleigh(2, 2).shape == (2, 2)
    assert rayleigh.generate_rayleigh(4, 2, (5,)).shape == (5, 4, 2)
    assert rayleigh.generate_rayleigh(4, 2, 5).shape == (5, 4, 2)
    assert rayleigh.generate_rayleigh(2, 4, (3, 7)).shape == (3, 7, 2, 4)


def test_entries_are_circularly_symmetric_unit_power() -> None:
    """E[h] = 0, E[|h|^2] = 1, E[Re(h)^2] = E[Im(h)^2] = 1/2, E[h^2] = 0."""
    rng = np.random.default_rng(SEED)
    samples = rayleigh.generate_rayleigh(2, 2, (250_000,), rng=rng)
    flat = samples.ravel()
    n_samples = flat.size
    real = flat.real
    imag = flat.imag

    mean_bound = 5.0 * np.sqrt(0.5 / n_samples)
    assert abs(real.mean()) < mean_bound
    assert abs(imag.mean()) < mean_bound

    variance_bound = 5.0 * np.sqrt(0.5 / n_samples)
    assert abs(real.var(ddof=1) - 0.5) < variance_bound
    assert abs(imag.var(ddof=1) - 0.5) < variance_bound

    power_bound = 5.0 / np.sqrt(n_samples)
    assert abs(np.mean(np.abs(flat) ** 2) - 1.0) < power_bound

    pseudo_covariance_bound = 5.0 * np.sqrt(2.0 / n_samples)
    assert abs(np.mean(flat**2)) < pseudo_covariance_bound

    cross_bound = 5.0 * 0.5 / np.sqrt(n_samples)
    assert abs(np.mean(real * imag)) < cross_bound


def test_amplitude_is_rayleigh() -> None:
    """|h| has Rayleigh density with E|h| = sqrt(pi)/2 and P(|h|<0.1)=1-e^-0.01."""
    rng = np.random.default_rng(SEED + 1)
    amplitude = np.abs(rayleigh.generate_rayleigh(2, 2, (250_000,), rng=rng)).ravel()
    n_samples = amplitude.size

    mean_bound = 5.0 * np.sqrt((2.0 - np.pi / 2.0) / 2.0 / n_samples)
    assert abs(amplitude.mean() - np.sqrt(np.pi) / 2.0) < mean_bound

    probability = 1.0 - np.exp(-0.01)
    probability_bound = 5.0 * np.sqrt(probability * (1.0 - probability) / n_samples)
    assert abs(np.mean(amplitude < 0.1) - probability) < probability_bound


@pytest.mark.parametrize(("nr", "nt"), [(2, 2), (4, 2), (4, 4)])
def test_channel_energy_is_nr_nt(nr: int, nt: int) -> None:
    """E[||H||_F^2] = Nr Nt, equation (3.3)."""
    n_channels = 100_000
    rng = np.random.default_rng(SEED + 2)
    channel = rayleigh.generate_rayleigh(nr, nt, (n_channels,), rng=rng)
    energy = np.sum(np.abs(channel) ** 2, axis=(1, 2))

    expected = float(nr * nt)
    standard_error = np.sqrt(expected / n_channels)
    assert abs(energy.mean() - expected) <= 5.0 * standard_error


def test_entries_are_independent() -> None:
    """E[h_i h_j^*] = delta_ij across the entries of one channel matrix."""
    n_channels = 200_000
    rng = np.random.default_rng(SEED + 3)
    entries = rayleigh.generate_rayleigh(2, 2, (n_channels,), rng=rng).reshape(
        n_channels, 4
    )
    gram = entries.T @ entries.conj() / n_channels

    bound = 5.0 / np.sqrt(n_channels)
    np.testing.assert_allclose(np.diag(gram), 1.0, rtol=0.0, atol=bound)
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off_diagonal)) < bound


def test_generation_is_reproducible_for_a_fixed_seed() -> None:
    first = rayleigh.generate_rayleigh(2, 2, (5,), rng=np.random.default_rng(SEED))
    second = rayleigh.generate_rayleigh(2, 2, (5,), rng=np.random.default_rng(SEED))
    other = rayleigh.generate_rayleigh(2, 2, (5,), rng=np.random.default_rng(SEED + 1))

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


@pytest.mark.parametrize(
    ("nr", "nt", "batch_shape"),
    [
        (0, 2, ()),
        (2, 0, ()),
        (-1, 2, ()),
        (2, -1, ()),
        (2.5, 2, ()),
        (2, 2, (-1,)),
        (2, 2, (1.5,)),
        (2, 2, 2.5),
    ],
)
def test_invalid_dimensions_are_rejected(
    nr: object, nt: object, batch_shape: object
) -> None:
    with pytest.raises(ValueError):
        rayleigh.generate_rayleigh(nr, nt, batch_shape)  # type: ignore[arg-type]
