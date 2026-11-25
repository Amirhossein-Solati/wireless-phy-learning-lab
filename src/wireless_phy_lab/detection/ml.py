"""Exhaustive maximum-likelihood detection for QPSK spatial multiplexing.

Equation (7.8): choose the candidate vector in ``S^Nt`` that minimises
``||y - H x||^2``. For QPSK the candidate set has ``4^Nt`` points: 16 at 2x2
and 256 at 4x4. The implementation loops over candidates and vectorises over
the batch, so memory stays ``O(batch * Nr)`` rather than
``O(batch * Nr * 4^Nt)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wireless_phy_lab.detection.linear import _as_channel_and_received
from wireless_phy_lab.modulation import qpsk

__all__ = ["ml_detect"]


def _qpsk_candidates(nt: int) -> NDArray[np.complex128]:
    """Return all ``4^nt`` QPSK candidate vectors, shape ``(4^nt, nt)``."""
    count = 4**nt
    indices = np.arange(count)
    candidates = np.empty((count, nt), dtype=np.complex128)
    for antenna in range(nt):
        candidates[:, antenna] = qpsk.CONSTELLATION[(indices // (4**antenna)) % 4]
    return candidates


def ml_detect(H: ArrayLike, y: ArrayLike) -> NDArray[np.complex128]:
    """Exhaustive ML detection for QPSK, equation (7.8).

    Parameters
    ----------
    H
        Channel array of shape ``(..., Nr, Nt)``.
    y
        Received array of shape ``(..., Nr)``.

    Returns
    -------
    numpy.ndarray
        Hard symbol estimates of shape ``(..., Nt)``, selected from the QPSK
        constellation. Convert to bits with ``qpsk.demap``.

    Notes
    -----
    Complexity is ``O(4^Nt Nr Nt)`` per channel use. This is the optimal
    reference for the MVP antenna sizes; it is not a scalable detector.
    """
    channel, received = _as_channel_and_received(H, y)
    nt = channel.shape[-1]
    batch_shape = channel.shape[:-2]

    best_metric = np.full(batch_shape, np.inf, dtype=np.float64)
    best = np.zeros(batch_shape + (nt,), dtype=np.complex128)

    for candidate in _qpsk_candidates(nt):
        residual = received - channel @ candidate
        metric = np.sum(np.abs(residual) ** 2, axis=-1)
        better = metric < best_metric
        best_metric = np.where(better, metric, best_metric)
        best = np.where(better[..., None], candidate, best)

    return best
