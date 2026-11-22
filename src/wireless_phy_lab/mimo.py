"""Spatial-multiplexing MIMO signal generation, equation (2.1).

The transmit vector for one channel use is

    x = sum_{j=1}^{Nt} s_j e_j,

so antenna ``j`` carries the independent symbol ``s_j``. Stacking ``Ns``
channel uses column-wise gives the transmit matrix ``X`` with shape
``(Nt, Ns)``. Every entry is one QPSK symbol with ``Es = E[|s|^2] = 1``.

Consequences that matter downstream:

- Each column has ``||x||^2 = Nt`` for QPSK, because every symbol has unit
  magnitude. Total transmit power therefore grows with ``Nt``; this is the
  SNR-convention trap in section 6 of the mathematical model.
- The streams are independent by construction, so
  ``E[s_j s_k^*] = delta_jk``. There is no precoding and no power allocation.

Shapes
------
- bits: ``(Nt, Ns, 2)``, last axis ``(b_I, b_Q)``
- symbols: ``(Nt, Ns)``, row ``j`` is stream ``j``
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wireless_phy_lab.modulation import qpsk

__all__ = [
    "generate_bit_streams",
    "generate_transmit_signal",
    "spatial_multiplex",
]


def _positive_int(value: int, name: str) -> int:
    """Validate a positive integer dimension."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def generate_bit_streams(
    nt: int,
    n_symbols: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.uint8]:
    """Generate independent equiprobable bits for ``nt`` transmit streams.

    Parameters
    ----------
    nt
        Number of transmit antennas, a positive integer.
    n_symbols
        Number of channel uses, a positive integer.
    rng
        NumPy generator. A fresh default generator is used when omitted, which
        is not reproducible; pass a seeded generator in experiments.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``(nt, n_symbols, 2)``. The last axis is
        ``(b_I, b_Q)``; every bit is an independent fair coin.
    """
    nt = _positive_int(nt, "nt")
    n_symbols = _positive_int(n_symbols, "n_symbols")

    if rng is None:
        rng = np.random.default_rng()

    return rng.integers(
        0,
        2,
        size=(nt, n_symbols, qpsk.BITS_PER_SYMBOL),
        dtype=np.uint8,
    )


def spatial_multiplex(bits: ArrayLike) -> NDArray[np.complex128]:
    """Map per-antenna bit streams to the transmit matrix ``X``.

    Parameters
    ----------
    bits
        Array of shape ``(Nt, Ns, 2)`` with values in ``{0, 1}``.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(Nt, Ns)``. Column ``k`` is the transmit
        vector for channel use ``k``.
    """
    bit_array = np.asarray(bits)
    if bit_array.ndim != 3 or bit_array.shape[-1] != qpsk.BITS_PER_SYMBOL:
        raise ValueError(
            f"bits must have shape (Nt, Ns, {qpsk.BITS_PER_SYMBOL}), "
            f"got {bit_array.shape}"
        )
    if bit_array.shape[0] < 1 or bit_array.shape[1] < 1:
        raise ValueError("Nt and Ns must be positive")

    return qpsk.map_bits(bit_array)


def generate_transmit_signal(
    nt: int,
    n_symbols: int,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.complex128]]:
    """Generate data bits and the corresponding spatial-multiplexed signal.

    This is a convenience composition of :func:`generate_bit_streams` and
    :func:`spatial_multiplex`. It returns both arrays so that a receiver can
    compare decisions against the transmitted bits without regenerating them.

    Returns
    -------
    tuple of numpy.ndarray
        ``(bits, symbols)`` with shapes ``(Nt, Ns, 2)`` and ``(Nt, Ns)``.
    """
    bits = generate_bit_streams(nt, n_symbols, rng=rng)
    return bits, spatial_multiplex(bits)
