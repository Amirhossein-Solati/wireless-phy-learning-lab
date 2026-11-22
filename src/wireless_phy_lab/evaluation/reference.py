"""Closed-form BER reference curves used to validate the Monte Carlo engine.

Equations (8.2) and (8.4) of the mathematical model:

- AWGN QPSK: ``Pb = Q(sqrt(2 Eb/N0))``.
- ZF over i.i.d. Rayleigh, QPSK: the exact finite-sum expression with
  ``m = Nr - Nt + 1`` and ``mu = sqrt(rho / (2 + rho))``, where ``rho = Es/N0``.

These are reference curves, not simulation output. The MMSE Gaussian
approximation is deliberately not included: it is a sanity band, not a
validation target.
"""

from __future__ import annotations

from math import comb

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erfc

__all__ = ["awgn_qpsk_ber", "qfunc", "zf_rayleigh_ber"]


def qfunc(x: ArrayLike) -> NDArray[np.float64]:
    """Gaussian tail probability ``Q(x) = erfc(x/sqrt(2))/2``."""
    return 0.5 * erfc(np.asarray(x, dtype=np.float64) / np.sqrt(2.0))


def awgn_qpsk_ber(eb_n0: ArrayLike) -> NDArray[np.float64]:
    """Uncoded QPSK BER over AWGN, equation (8.2).

    Parameters
    ----------
    eb_n0
        Linear ``Eb/N0`` ratio, non-negative.

    Returns
    -------
    numpy.ndarray
        ``Q(sqrt(2 Eb/N0))``, preserving the input shape.
    """
    ratio = np.asarray(eb_n0, dtype=np.float64)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ValueError("Eb/N0 must be finite and non-negative")
    return qfunc(np.sqrt(2.0 * ratio))


def zf_rayleigh_ber(
    es_n0: ArrayLike,
    diversity_order: int,
) -> NDArray[np.float64]:
    """Exact ZF BER over i.i.d. Rayleigh for QPSK, equation (8.4).

    Parameters
    ----------
    es_n0
        Linear per-stream ``Es/N0`` ratio.
    diversity_order
        ``m = Nr - Nt + 1``, a positive integer. For square arrays ``m = 1``.

    Returns
    -------
    numpy.ndarray
        Exact average BER, preserving the input shape.
    """
    if (
        isinstance(diversity_order, bool)
        or not isinstance(diversity_order, (int, np.integer))
        or diversity_order < 1
    ):
        raise ValueError("diversity_order must be a positive integer")

    rho = np.asarray(es_n0, dtype=np.float64)
    if np.any(~np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("Es/N0 must be finite and non-negative")

    m = int(diversity_order)
    mu = np.sqrt(rho / (2.0 + rho))
    series = np.zeros_like(rho)
    for k in range(m):
        series = series + comb(m - 1 + k, k) * ((1.0 + mu) / 2.0) ** k

    return ((1.0 - mu) / 2.0) ** m * series
