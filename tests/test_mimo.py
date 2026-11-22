"""Tests for spatial-multiplexing MIMO signal generation, equation (2.1).

Seed policy: every Monte Carlo test fixes its seed. Same seed, same bits.
"""

from __future__ import annotations

import numpy as np
import pytest

from wireless_phy_lab import mimo
from wireless_phy_lab.channel import awgn, rayleigh
from wireless_phy_lab.modulation import qpsk

SEED = 20261125


def test_bit_streams_shape_dtype_and_balance() -> None:
    n_symbols = 100_000
    rng = np.random.default_rng(SEED)
    bits = mimo.generate_bit_streams(2, n_symbols, rng=rng)

    assert bits.shape == (2, n_symbols, qpsk.BITS_PER_SYMBOL)
    assert bits.dtype == np.uint8
    assert set(np.unique(bits)) <= {0, 1}

    n_bits = bits.size
    standard_error = np.sqrt(0.25 / n_bits)
    assert abs(float(bits.mean()) - 0.5) <= 5.0 * standard_error
    per_stream = bits.mean(axis=(1, 2))
    per_stream_error = np.sqrt(0.25 / (n_symbols * qpsk.BITS_PER_SYMBOL))
    assert np.all(np.abs(per_stream - 0.5) <= 5.0 * per_stream_error)


def test_bit_streams_are_reproducible_for_a_fixed_seed() -> None:
    first = mimo.generate_bit_streams(3, 20, rng=np.random.default_rng(SEED))
    second = mimo.generate_bit_streams(3, 20, rng=np.random.default_rng(SEED))
    other = mimo.generate_bit_streams(3, 20, rng=np.random.default_rng(SEED + 1))

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_spatial_multiplex_maps_to_unit_energy_qpsk() -> None:
    rng = np.random.default_rng(SEED + 1)
    bits = mimo.generate_bit_streams(2, 1_000, rng=rng)
    symbols = mimo.spatial_multiplex(bits)

    assert symbols.shape == (2, 1_000)
    assert symbols.dtype == np.complex128
    assert np.all(np.isin(symbols, qpsk.CONSTELLATION))
    np.testing.assert_array_equal(qpsk.demap(symbols), bits)

    np.testing.assert_allclose(np.abs(symbols) ** 2, 1.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(np.mean(np.abs(symbols) ** 2), 1.0, rtol=0.0, atol=1e-15)

    # Every channel-use column carries Nt unit-energy symbols.
    column_energy = np.sum(np.abs(symbols) ** 2, axis=0)
    np.testing.assert_allclose(column_energy, 2.0, rtol=0.0, atol=1e-14)


def test_streams_are_uncorrelated() -> None:
    """E[s_j s_k^*] = delta_jk for independent streams, equation (2.1)."""
    nt = 4
    n_symbols = 200_000
    rng = np.random.default_rng(SEED + 2)
    bits = mimo.generate_bit_streams(nt, n_symbols, rng=rng)
    symbols = mimo.spatial_multiplex(bits)

    gram = symbols @ symbols.conj().T / n_symbols
    bound = 5.0 / np.sqrt(n_symbols)
    np.testing.assert_allclose(np.diag(gram), 1.0, rtol=0.0, atol=bound)
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off_diagonal)) < bound


def test_generate_transmit_signal_is_the_two_step_composition() -> None:
    bits, symbols = mimo.generate_transmit_signal(
        3, 17, rng=np.random.default_rng(SEED)
    )
    reference_bits = mimo.generate_bit_streams(3, 17, rng=np.random.default_rng(SEED))
    reference_symbols = mimo.spatial_multiplex(reference_bits)

    assert bits.shape == (3, 17, qpsk.BITS_PER_SYMBOL)
    assert symbols.shape == (3, 17)
    np.testing.assert_array_equal(bits, reference_bits)
    np.testing.assert_array_equal(symbols, reference_symbols)


@pytest.mark.parametrize(
    "bits",
    [
        np.zeros(2),
        np.zeros((2, 2)),
        np.zeros((2, 2, 3)),
        np.zeros((0, 2, 2)),
        np.zeros((2, 0, 2)),
        np.full((2, 2, 2), 2),
    ],
)
def test_spatial_multiplex_rejects_invalid_bits(bits: np.ndarray) -> None:
    with pytest.raises(ValueError):
        mimo.spatial_multiplex(bits)


@pytest.mark.parametrize(
    ("nt", "n_symbols"),
    [(0, 10), (2, 0), (-1, 10), (2, -1), (2.0, 10), (2, 10.0)],
)
def test_invalid_sizes_are_rejected(nt: object, n_symbols: object) -> None:
    with pytest.raises(ValueError):
        mimo.generate_bit_streams(nt, n_symbols)  # type: ignore[arg-type]


def test_components_compose_into_a_received_signal() -> None:
    """The four components produce y = Hx + n without any detector.

    Shape conventions: H is one matrix per channel use, x has one column per
    channel use, and noise has one row per channel use.
    """
    nt, nr, n_symbols = 2, 2, 1_000
    es_n0 = 4.0
    rng = np.random.default_rng(SEED + 3)

    bits, symbols = mimo.generate_transmit_signal(nt, n_symbols, rng=rng)
    channel = rayleigh.generate_rayleigh(nr, nt, (n_symbols,), rng=rng)
    noise = awgn.generate_noise((n_symbols, nr), es_n0, rng=rng)
    received = np.einsum("sij,js->si", channel, symbols) + noise

    assert received.shape == (n_symbols, nr)
    assert np.all(np.isfinite(received))

    k = 17
    np.testing.assert_allclose(
        received[k], channel[k] @ symbols[:, k] + noise[k], rtol=0.0, atol=1e-14
    )

    # E|y_i|^2 = Nt Es + sigma^2 because each column has Nt unit-energy symbols.
    expected_power = nt + 1.0 / es_n0
    standard_error = expected_power / np.sqrt(n_symbols * nr)
    assert abs(np.mean(np.abs(received) ** 2) - expected_power) <= 5.0 * standard_error

    assert bits.shape == (nt, n_symbols, qpsk.BITS_PER_SYMBOL)
