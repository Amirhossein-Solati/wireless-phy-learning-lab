"""Gray-mapped QPSK with unit average symbol energy.

The constellation follows the approved mathematical model, equations
(5.1)-(5.3):

    S = {(+/-1 +/- j) / sqrt(2)},   M = 4,   Es = E[|s|^2] = 1.

Each symbol carries two bits. ``b_I`` selects the sign of the in-phase rail and
``b_Q`` the sign of the quadrature rail, so the labelling is Gray: any two
neighbouring points differ in exactly one bit.

Energy convention
-----------------
QPSK has ``Es = (log2 M) Eb = 2 Eb``, i.e. ``Es/Eb = bits per symbol``. The
relation is defined once, in ``ES_OVER_EB``, and every conversion between the
two ratios goes through ``es_n0_from_eb_n0`` or ``eb_n0_from_es_n0``. No other
module should hard-code the factor of two.

Units
-----
After normalisation the model is dimensionless: ``Es = 1`` and the complex noise
variance for a target per-stream ``Es/N0`` is ``1 / es_n0``. For QPSK, a noise
variance ``sigma^2`` per complex dimension corresponds to ``sigma^2 / 2`` per
real dimension.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "BITS_PER_SYMBOL",
    "CONSTELLATION",
    "ES_OVER_EB",
    "demap",
    "eb_n0_from_es_n0",
    "es_n0_from_eb_n0",
    "map_bits",
]

BITS_PER_SYMBOL: Final[int] = 2
"""Number of bits carried by one QPSK symbol, ``log2 M``."""

ES_OVER_EB: Final[float] = float(BITS_PER_SYMBOL)
"""Symbol-to-bit energy ratio, ``Es/Eb = 2`` for QPSK. Defined here only."""

# Bit labels in the order (b_I, b_Q) = (0,0), (0,1), (1,0), (1,1). A QPSK
# symbol is two independent BPSK rails, so this ordering is Gray by
# construction: neighbouring points differ in one rail, hence one bit.
_LABELS: Final[NDArray[np.uint8]] = np.array(
    [[0, 0], [0, 1], [1, 0], [1, 1]],
    dtype=np.uint8,
)


def _as_bits(bits: ArrayLike) -> NDArray[np.int64]:
    """Validate a bit array whose last axis holds one bit per symbol dimension."""
    array = np.asarray(bits)
    if array.ndim < 1 or array.shape[-1] != BITS_PER_SYMBOL:
        raise ValueError(
            f"bits must have shape (..., {BITS_PER_SYMBOL}), got {array.shape}"
        )
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("bits must contain only 0 and 1")
    return array.astype(np.int64, copy=False)


def map_bits(bits: ArrayLike) -> NDArray[np.complex128]:
    """Map pairs of bits to Gray-mapped QPSK symbols.

    Parameters
    ----------
    bits
        Array of shape ``(..., 2)``. ``bits[..., 0]`` is the in-phase bit and
        ``bits[..., 1]`` the quadrature bit. A zero selects the positive rail
        and a one the negative rail.

    Returns
    -------
    numpy.ndarray
        Complex symbols of shape ``bits.shape[:-1]``, with average energy one
        under uniformly distributed bits.
    """
    bit_array = _as_bits(bits)
    in_phase = 1.0 - 2.0 * bit_array[..., 0]
    quadrature = 1.0 - 2.0 * bit_array[..., 1]
    return (in_phase + 1j * quadrature) / np.sqrt(2.0)


CONSTELLATION: Final[NDArray[np.complex128]] = map_bits(_LABELS)
CONSTELLATION.setflags(write=False)
"""The four QPSK points with unit average energy, ordered as ``_LABELS``."""


def demap(symbols: ArrayLike) -> NDArray[np.uint8]:
    """Hard-decision demap received symbols to bit pairs.

    For QPSK the nearest-neighbour rule decouples into one threshold per rail:
    the sign of the real part decides ``b_I`` and the sign of the imaginary
    part decides ``b_Q``. A coordinate exactly on the decision boundary is
    assigned to the positive rail, i.e. bit 0.

    Parameters
    ----------
    symbols
        Complex received values of arbitrary shape.

    Returns
    -------
    numpy.ndarray
        ``uint8`` bit array of shape ``symbols.shape + (2,)``.
    """
    received = np.asarray(symbols, dtype=np.complex128)
    bits = np.empty(received.shape + (BITS_PER_SYMBOL,), dtype=np.uint8)
    bits[..., 0] = received.real < 0.0
    bits[..., 1] = received.imag < 0.0
    return bits


def es_n0_from_eb_n0(eb_n0: ArrayLike) -> NDArray[np.float64]:
    """Convert linear ``Eb/N0`` to linear ``Es/N0``.

    Parameters
    ----------
    eb_n0
        Energy per bit over noise power spectral density, as a linear ratio.

    Returns
    -------
    numpy.ndarray
        ``Es/N0 = 2 Eb/N0`` for QPSK, preserving the input shape.
    """
    return ES_OVER_EB * np.asarray(eb_n0, dtype=np.float64)


def eb_n0_from_es_n0(es_n0: ArrayLike) -> NDArray[np.float64]:
    """Convert linear ``Es/N0`` to linear ``Eb/N0``, the inverse of (6.2)."""
    return np.asarray(es_n0, dtype=np.float64) / ES_OVER_EB
