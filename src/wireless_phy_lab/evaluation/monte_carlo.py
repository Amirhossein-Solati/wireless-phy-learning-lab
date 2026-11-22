"""Seeded Monte Carlo BER simulation for the QPSK MIMO baseline.

One point of a BER curve is simulated here: draw bits, map them to a
spatial-multiplexed frame, draw an independent Rayleigh channel per channel
use, add calibrated AWGN, run the supplied detector, demap, and count bit
errors. The loop stops at a target error count or a maximum number of bits.

Seed policy
-----------
The generator is ``numpy.random.default_rng(seed)``. All random draws (bits,
channel, noise) come from it in a fixed order, so the same seed and parameters
reproduce the same result. The detector itself consumes no randomness.

Stopping rule
-------------
Stop when ``errors >= target_errors`` or when ``bits >= max_bits``. The latter
is a safety limit, not a target. The reported interval always reflects the
number of bits actually simulated.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from wireless_phy_lab import mimo
from wireless_phy_lab.channel import awgn, rayleigh
from wireless_phy_lab.evaluation.ber import (
    BerPoint,
    ber_estimate,
    count_bit_errors,
)
from wireless_phy_lab.modulation import qpsk

__all__ = ["Detector", "simulate_ber_point"]

Detector = Callable[
    [NDArray[np.complex128], NDArray[np.complex128], float],
    NDArray[np.complex128],
]


def _positive_int(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def simulate_ber_point(
    detector: Detector,
    *,
    nr: int,
    nt: int,
    eb_n0_db: float,
    seed: int,
    target_errors: int = 2000,
    max_bits: int = 8_000_000,
    batch_symbols: int = 20_000,
    confidence: float = 0.95,
) -> BerPoint:
    """Simulate one BER point for uncoded QPSK over i.i.d. Rayleigh fading.

    Parameters
    ----------
    detector
        Callable ``detector(H, y, noise_variance) -> x_hat``. ``H`` has shape
        ``(batch, Nr, Nt)``, ``y`` has shape ``(batch, Nr)``, and the return
        value has shape ``(batch, Nt)``. Wrap ZF as
        ``lambda H, y, sigma2: zf_detect(H, y)``.
    nr, nt
        Receive and transmit antenna counts.
    eb_n0_db
        Per-stream ``Eb/N0`` in dB. The x-axis convention is decision D-01.
    seed
        Non-negative seed for the generator.
    target_errors
        Error count at which the simulation stops.
    max_bits
        Safety limit on the number of simulated bits.
    batch_symbols
        Channel uses per batch. A large batch amortises Python overhead.
    confidence
        Confidence level for the Wilson interval.

    Returns
    -------
    BerPoint
        Aggregate BER and one estimate per stream, each with a Wilson interval.
        Decision D-07 requires both.
    """
    nr = _positive_int(nr, "nr")
    nt = _positive_int(nt, "nt")
    target_errors = _positive_int(target_errors, "target_errors")
    batch_symbols = _positive_int(batch_symbols, "batch_symbols")
    max_bits = _positive_int(max_bits, "max_bits")

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if max_bits < nt * qpsk.BITS_PER_SYMBOL:
        raise ValueError("max_bits must allow at least one QPSK symbol per stream")
    if not np.isfinite(eb_n0_db):
        raise ValueError("eb_n0_db must be finite")

    eb_n0 = 10.0 ** (eb_n0_db / 10.0)
    es_n0 = float(qpsk.es_n0_from_eb_n0(eb_n0))
    noise_variance = 1.0 / es_n0

    rng = np.random.default_rng(int(seed))
    errors = 0
    total_bits = 0
    stream_errors = np.zeros(nt, dtype=np.int64)
    stream_bits = np.zeros(nt, dtype=np.int64)

    while errors < target_errors and total_bits < max_bits:
        remaining_bits = max_bits - total_bits
        max_symbols = remaining_bits // (nt * qpsk.BITS_PER_SYMBOL)
        batch = min(batch_symbols, int(max_symbols))
        if batch < 1:
            break

        bits = mimo.generate_bit_streams(nt, batch, rng=rng)
        symbols = mimo.spatial_multiplex(bits)
        channel = rayleigh.generate_rayleigh(nr, nt, (batch,), rng=rng)
        noise = awgn.generate_noise((batch, nr), es_n0, rng=rng)
        received = np.einsum("sij,js->si", channel, symbols) + noise

        estimates = detector(channel, received, noise_variance)
        if estimates.shape != (batch, nt):
            raise ValueError(
                f"detector returned {estimates.shape}, expected {(batch, nt)}"
            )

        estimated_bits = qpsk.demap(estimates)
        transmitted_bits = np.transpose(bits, (1, 0, 2))
        mismatch = transmitted_bits != estimated_bits
        stream_errors += np.count_nonzero(mismatch, axis=(0, 2))
        stream_bits += transmitted_bits.shape[0] * qpsk.BITS_PER_SYMBOL
        errors += count_bit_errors(transmitted_bits, estimated_bits)
        total_bits += transmitted_bits.size

    aggregate = ber_estimate(errors, total_bits, confidence=confidence)
    per_stream = tuple(
        ber_estimate(
            int(stream_errors[stream]),
            int(stream_bits[stream]),
            confidence=confidence,
        )
        for stream in range(nt)
    )
    return BerPoint(aggregate=aggregate, per_stream=per_stream)
