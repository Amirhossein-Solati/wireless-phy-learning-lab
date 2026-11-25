"""Linear MIMO detectors: zero forcing and biased MMSE.

Model reference: equations (7.1)-(7.7). After normalisation ``Es = 1``, so the
noise variance ``sigma^2`` is also ``1/(Es/N0)``.

Shapes
------
``H`` is ``(..., Nr, Nt)`` and ``y`` is ``(..., Nr)``, with matching leading
batch dimensions. The returned symbol estimates are ``(..., Nt)``. A single
channel use has no leading dimension.

Numerics
--------
Both detectors solve the normal equations directly and never form
``(H^H H)^{-1}``. ZF squares the condition number of ``H``, so report
``condition_number(H)`` alongside ZF results and treat ill-conditioned channels
with care.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["condition_number", "mmse_detect", "zf_detect"]


def _as_channel(H: ArrayLike) -> NDArray[np.complex128]:
    """Validate a channel array of shape ``(..., Nr, Nt)``."""
    channel = np.asarray(H, dtype=np.complex128)
    if channel.ndim < 2:
        raise ValueError(f"H must have shape (..., Nr, Nt), got {channel.shape}")
    if channel.shape[-2] < 1 or channel.shape[-1] < 1:
        raise ValueError("Nr and Nt must be positive")
    return channel


def _as_channel_and_received(
    H: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Validate the channel and received arrays, including batch agreement."""
    channel = _as_channel(H)
    received = np.asarray(y, dtype=np.complex128)

    if received.ndim != channel.ndim - 1:
        raise ValueError(
            f"y must have shape (..., Nr) matching H; got y {received.shape} "
            f"and H {channel.shape}"
        )
    if channel.shape[:-2] != received.shape[:-1]:
        raise ValueError(
            f"batch dimensions of H {channel.shape[:-2]} and y "
            f"{received.shape[:-1]} must match"
        )
    if channel.shape[-2] != received.shape[-1]:
        raise ValueError(
            f"H has Nr={channel.shape[-2]} but y has {received.shape[-1]} entries"
        )
    return channel, received


def _as_noise_variance(noise_variance: float) -> float:
    """Validate a scalar complex noise variance."""
    try:
        variance = float(noise_variance)
    except (TypeError, ValueError) as exc:
        raise ValueError("noise_variance must be a real scalar") from exc
    if not np.isfinite(variance) or variance < 0.0:
        raise ValueError("noise_variance must be finite and non-negative")
    return variance


def _solve_normal_equations(
    matrix: NDArray[np.complex128], rhs: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Solve ``matrix x = rhs`` directly, without forming an inverse."""
    return np.linalg.solve(matrix, rhs[..., None])[..., 0]


def condition_number(H: ArrayLike) -> NDArray[np.float64]:
    """Return the 2-norm condition number of each channel matrix.

    Parameters
    ----------
    H
        Channel array of shape ``(..., Nr, Nt)``.

    Returns
    -------
    numpy.ndarray
        Condition number with shape ``H.shape[:-2]``. Report it with any ZF
        result: ZF amplifies noise by ``1/sigma_min^2``.
    """
    return np.asarray(np.linalg.cond(_as_channel(H)), dtype=np.float64)


def zf_detect(H: ArrayLike, y: ArrayLike) -> NDArray[np.complex128]:
    """Zero-forcing detection, equations (7.1)-(7.3).

    Solves the normal equations

    .. math:: (H^H H) x_hat = H^H y

    with a direct linear solve. No explicit inverse is formed.

    Parameters
    ----------
    H
        Channel array of shape ``(..., Nr, Nt)``. Requires ``Nr >= Nt`` and
        full column rank.
    y
        Received array of shape ``(..., Nr)``.

    Returns
    -------
    numpy.ndarray
        Symbol estimates of shape ``(..., Nt)``. Convert to QPSK bits with
        ``qpsk.demap``.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or ``Nr < Nt``.
    numpy.linalg.LinAlgError
        If ``H`` does not have full column rank, in which case ZF is undefined.
    """
    channel, received = _as_channel_and_received(H, y)
    nr, nt = channel.shape[-2:]

    if nr < nt:
        raise ValueError(f"ZF requires Nr >= Nt (A4); got Nr={nr}, Nt={nt}")

    gram = channel.conj().swapaxes(-2, -1) @ channel
    matched = np.einsum("...ij,...i->...j", channel.conj(), received)
    return _solve_normal_equations(gram, matched)


def mmse_detect(
    H: ArrayLike, y: ArrayLike, noise_variance: float
) -> NDArray[np.complex128]:
    """Biased MMSE detection, equations (7.4)-(7.7), with ``Es = 1``.

    Solves

    .. math:: (H^H H + sigma^2 I) x_hat = H^H y

    with a direct linear solve. No explicit inverse is formed. This is the
    biased variant of decision D-05.
    For QPSK hard decisions the bias is irrelevant: the decision thresholds are
    at zero and the bias is real. It is positive for a full-rank channel and
    zero for a null stream.

    Parameters
    ----------
    H
        Channel array of shape ``(..., Nr, Nt)``.
    y
        Received array of shape ``(..., Nr)``.
    noise_variance
        Complex noise variance ``sigma^2 = 1/(Es/N0)``, finite and
        non-negative. It is not an SNR in dB and not ``Eb/N0``.

    Returns
    -------
    numpy.ndarray
        Symbol estimates of shape ``(..., Nt)``. Convert to QPSK bits with
        ``qpsk.demap``.

    Notes
    -----
    A zero variance reduces the estimator to ZF exactly. Unlike ZF, the
    regularised Gram matrix is positive definite for any positive variance,
    including when ``Nr < Nt``.
    """
    channel, received = _as_channel_and_received(H, y)
    variance = _as_noise_variance(noise_variance)
    nt = channel.shape[-1]

    gram = channel.conj().swapaxes(-2, -1) @ channel
    gram = gram + variance * np.eye(nt, dtype=np.complex128)
    matched = np.einsum("...ij,...i->...j", channel.conj(), received)
    return _solve_normal_equations(gram, matched)
