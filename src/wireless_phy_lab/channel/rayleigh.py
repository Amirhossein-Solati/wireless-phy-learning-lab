"""Flat Rayleigh fading with i.i.d. unit-power entries.

The model is equations (3.1)-(3.3):

    h_ij ~ CN(0, 1) i.i.d.,   E[|h_ij|^2] = 1,   E[h_ij^2] = 0,

with ``|h_ij|`` Rayleigh distributed and ``angle(h_ij)`` uniform. A circularly
symmetric complex Gaussian sample is generated as ``(X + jY)/sqrt(2)`` with
independent standard normal ``X`` and ``Y``.

Fading dynamics follow decision D-02: ergodic, i.i.d. per channel use. The
``batch_shape`` argument produces independent matrices for a batch of channel
uses. There is no correlation, no path loss, and no keyhole.

Units are dimensionless: the path loss has been normalised away, so a BER curve
computed with this generator is a function of ``Es/N0`` only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["generate_rayleigh"]


def _positive_int(value: int, name: str) -> int:
    """Validate a positive integer dimension."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _as_batch_shape(batch_shape: int | tuple[int, ...]) -> tuple[int, ...]:
    """Normalise and validate the leading batch dimensions."""
    if isinstance(batch_shape, (int, np.integer)) and not isinstance(batch_shape, bool):
        batch = (int(batch_shape),)
    else:
        try:
            batch = tuple(batch_shape)
        except TypeError as exc:
            raise ValueError("batch_shape must be an int or a tuple of ints") from exc

    for dim in batch:
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim < 0:
            raise ValueError("batch_shape entries must be non-negative integers")
    return tuple(int(dim) for dim in batch)


def generate_rayleigh(
    nr: int,
    nt: int,
    batch_shape: int | tuple[int, ...] = (),
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Generate i.i.d. ``CN(0, 1)`` Rayleigh channel matrices.

    Parameters
    ----------
    nr
        Number of receive antennas, a positive integer.
    nt
        Number of transmit antennas, a positive integer.
    batch_shape
        Leading dimensions for independent channel uses. The default ``()``
        returns one ``(nr, nt)`` matrix. Pass ``(n_channels,)`` for one
        independent matrix per channel use, or an integer as shorthand.
    rng
        NumPy generator. A fresh default generator is used when omitted, which
        is not reproducible; pass a seeded generator in experiments.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``batch_shape + (nr, nt)``. Each entry has
        ``E[|h|^2] = 1``, and ``E[||H||_F^2] = nr * nt`` per matrix.
    """
    nr = _positive_int(nr, "nr")
    nt = _positive_int(nt, "nt")
    batch = _as_batch_shape(batch_shape)

    if rng is None:
        rng = np.random.default_rng()

    shape = batch + (nr, nt)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (real + 1.0j * imag) / np.sqrt(2.0)
