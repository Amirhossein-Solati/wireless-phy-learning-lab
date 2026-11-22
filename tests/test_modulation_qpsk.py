"""Tests for Gray-mapped QPSK: equations (5.1)-(5.3), (6.2) and (8.2).

Seed policy: every Monte Carlo test fixes its seed here, and the seed is part of
the test contract. A change to the seed is a change to the experiment.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.special import erfc

from wireless_phy_lab.modulation import qpsk

SEED = 20261122


def qfunc(x: np.ndarray | float) -> np.ndarray | float:
    """Gaussian tail probability, ``Q(x) = erfc(x / sqrt(2)) / 2`` (DLMF 7.2)."""
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2.0))


def all_bit_labels() -> np.ndarray:
    """The four QPSK labels in the constellation order used by the mapper."""
    return np.array(list(itertools.product((0, 1), repeat=qpsk.BITS_PER_SYMBOL)), dtype=np.uint8)


def test_qfunc_matches_reference_anchors() -> None:
    """Test oracle guard: Q(3) must be 1.349898031630096e-3, not erfc(3)."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.array(
        [
            1.586552539314571e-1,
            2.275013194817922e-2,
            1.349898031630096e-3,
            3.167124183311998e-5,
        ]
    )
    np.testing.assert_allclose(qfunc(x), expected, rtol=1e-12, atol=0.0)


def test_constellation_is_unit_average_energy() -> None:
    """E[|s|^2] = 1 over the four equiprobable points, and every |s|^2 = 1."""
    points = qpsk.CONSTELLATION
    assert points.shape == (4,)
    np.testing.assert_allclose(np.abs(points) ** 2, 1.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(np.mean(np.abs(points) ** 2), 1.0, rtol=0.0, atol=1e-15)

    expected = np.array([1.0 + 1.0j, 1.0 - 1.0j, -1.0 + 1.0j, -1.0 - 1.0j]) / np.sqrt(2.0)
    np.testing.assert_allclose(points, expected, rtol=0.0, atol=1e-15)


def test_mapped_energy_is_one_for_uniform_bits() -> None:
    """The mapper, not just the constant table, produces unit average energy."""
    rng = np.random.default_rng(SEED)
    bits = rng.integers(0, 2, size=(200_000, qpsk.BITS_PER_SYMBOL), dtype=np.uint8)
    symbols = qpsk.map_bits(bits)

    np.testing.assert_allclose(np.abs(symbols) ** 2, 1.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(np.mean(np.abs(symbols) ** 2), 1.0, rtol=0.0, atol=1e-15)
    assert set(np.unique(symbols)) == set(qpsk.CONSTELLATION)


def test_gray_labels_round_trip() -> None:
    """demap(map(bits)) = bits for every label and for arbitrary bit arrays."""
    labels = all_bit_labels()
    np.testing.assert_array_equal(qpsk.demap(qpsk.map_bits(labels)), labels)

    rng = np.random.default_rng(SEED + 1)
    bits = rng.integers(0, 2, size=(3, 5, qpsk.BITS_PER_SYMBOL), dtype=np.uint8)
    symbols = qpsk.map_bits(bits)
    assert symbols.shape == bits.shape[:-1]

    decoded = qpsk.demap(symbols)
    assert decoded.shape == bits.shape
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, bits)


def test_gray_mapping_adjacent_points_differ_in_one_bit() -> None:
    """Nearest neighbours are one bit apart; diagonals are two bits apart."""
    labels = all_bit_labels()
    points = qpsk.map_bits(labels)
    distances = np.abs(points[:, None] - points[None, :])
    hamming = np.sum(labels[:, None, :] != labels[None, :, :], axis=-1)

    minimum_distance = np.sqrt(2.0)
    np.testing.assert_allclose(distances[distances > 0].min(), minimum_distance, rtol=1e-15)

    adjacent_pairs = 0
    for i in range(labels.shape[0]):
        for j in range(i + 1, labels.shape[0]):
            if np.isclose(distances[i, j], minimum_distance):
                adjacent_pairs += 1
                assert hamming[i, j] == 1
            else:
                assert hamming[i, j] == 2
    assert adjacent_pairs == 4


def test_demapper_is_nearest_neighbour() -> None:
    """Every decoded point is a nearest constellation point, ties included."""
    grid = np.linspace(-1.75, 1.75, 29)
    real, imag = np.meshgrid(grid, grid)
    received = (real + 1j * imag).ravel()

    decoded_points = qpsk.map_bits(qpsk.demap(received))
    distance_to_decoded = np.abs(received - decoded_points)
    distance_to_nearest = np.min(
        np.abs(received[:, None] - qpsk.CONSTELLATION[None, :]), axis=1
    )
    np.testing.assert_allclose(distance_to_decoded, distance_to_nearest, rtol=0.0, atol=1e-12)


def test_es_over_eb_is_two_in_one_place() -> None:
    """Es = 2 Eb for QPSK; the conversion is a round trip."""
    assert qpsk.BITS_PER_SYMBOL == 2
    assert qpsk.ES_OVER_EB == 2.0

    eb_n0 = np.array([0.0, 0.5, 10.0])
    np.testing.assert_array_equal(qpsk.es_n0_from_eb_n0(eb_n0), 2.0 * eb_n0)
    np.testing.assert_allclose(
        qpsk.eb_n0_from_es_n0(qpsk.es_n0_from_eb_n0(eb_n0)), eb_n0, rtol=1e-15, atol=0.0
    )


def test_map_rejects_malformed_bits() -> None:
    with pytest.raises(ValueError):
        qpsk.map_bits(np.array([0, 1, 0]))
    with pytest.raises(ValueError):
        qpsk.map_bits(np.array(0))
    with pytest.raises(ValueError):
        qpsk.map_bits(np.array([[0, 2]]))


@pytest.mark.parametrize("eb_n0_db", [0.0, 3.0, 6.0, 9.0])
def test_awgn_bit_error_rate_matches_closed_form(eb_n0_db: float) -> None:
    """AWGN QPSK BER = Q(sqrt(Es/N0)) = Q(sqrt(2 Eb/N0)), equation (8.2).

    The whole chain is exercised: mapper, unit-energy constellation, noise
    scaling at the requested Eb/N0, hard-decision demapper, error counter.
    """
    rng = np.random.default_rng(SEED + int(10 * eb_n0_db))
    n_bits = 1_000_000

    eb_n0 = 10.0 ** (eb_n0_db / 10.0)
    es_n0 = float(qpsk.es_n0_from_eb_n0(eb_n0))
    noise_variance = 1.0 / es_n0

    bits = rng.integers(
        0, 2, size=(n_bits // qpsk.BITS_PER_SYMBOL, qpsk.BITS_PER_SYMBOL), dtype=np.uint8
    )
    symbols = qpsk.map_bits(bits)
    noise = np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal(symbols.shape) + 1.0j * rng.standard_normal(symbols.shape)
    )

    decoded = qpsk.demap(symbols + noise)
    measured = float(np.mean(decoded != bits))
    expected = float(qfunc(np.sqrt(es_n0)))
    standard_error = np.sqrt(expected * (1.0 - expected) / n_bits)

    assert abs(measured - expected) <= 5.0 * standard_error, (
        f"Eb/N0={eb_n0_db} dB: measured {measured:.6e}, "
        f"closed form {expected:.6e}, standard error {standard_error:.2e}"
    )
