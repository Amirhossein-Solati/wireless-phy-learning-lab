"""Complex additive white Gaussian noise with SNR calibration.

The noise follows the approved model, equations (2.3), (4.1), (4.2) and (6.1).
After the unit-energy normalisation the model is dimensionless with ``Es = 1``,
so the per-receive-antenna noise is

    n ~ CN(0, sigma^2 I),   sigma^2 = N0/Es = 1/(Es/N0).

A circularly symmetric complex Gaussian sample splits into two independent real
Gaussians with variance ``sigma^2/2`` each. That factor of one half is the
complex-to-real split, not the ``Es = 2Eb`` relation: the latter is specific to
the modulation and lives in :mod:`wireless_phy_lab.modulation.qpsk`.

This module is modulation-agnostic. It takes a linear ``Es/N0`` ratio; callers
that work in ``Eb/N0`` convert with ``qpsk.es_n0_from_eb_n0`` first.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["add_awgn", "generate_noise", "noise_variance"]


def noise_variance(es_n0: ArrayLike) -> NDArray[np.float64]:
    """Complex noise variance for a linear ``Es/N0`` ratio.

    Parameters
    ----------
    es_n0
        Per-stream symbol energy over noise power spectral density, linear and
        strictly positive. dB values must be converted before calling.

    Returns
    -------
    numpy.ndarray
        ``sigma^2 = 1/(Es/N0)``, preserving the input shape.
    """
    ratio = np.asarray(es_n0, dtype=np.float64)
    if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0.0):
        raise ValueError("Es/N0 must be finite and strictly positive")
    return 1.0 / ratio


def generate_noise(
    shape: int | tuple[int, ...],
    es_n0: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Draw circularly symmetric complex AWGN of the requested shape.

    Parameters
    ----------
    shape
        Output shape, for example ``(Nr, Ns)`` for one channel matrix per
        column.
    es_n0
        Linear per-stream ``Es/N0``. The complex variance is ``1/es_n0``.
    rng
        NumPy generator. A fresh default generator is used when omitted, which
        is not reproducible; pass a seeded generator in experiments.

    Returns
    -------
    numpy.ndarray
        Complex noise samples with ``E[|n|^2] = 1/es_n0`` and
        ``E[n^2] = 0``.
    """
    variance = float(noise_variance(es_n0))
    if rng is None:
        rng = np.random.default_rng()
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return np.sqrt(variance / 2.0) * (real + 1.0j * imag)


def add_awgn(
    signal: ArrayLike,
    es_n0: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Add calibrated complex AWGN to a signal.

    Parameters
    ----------
    signal
        Complex samples, typically ``H x + n`` without the noise term.
    es_n0
        Linear per-stream ``Es/N0`` of the signal.
    rng
        NumPy generator, as in :func:`generate_noise`.

    Returns
    -------
    numpy.ndarray
        ``signal + noise`` with the same shape. The input is not modified.
    """
    samples = np.asarray(signal, dtype=np.complex128)
    return samples + generate_noise(samples.shape, es_n0, rng=rng)
