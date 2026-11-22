"""Bit-error counting and binomial confidence intervals.

The Monte Carlo BER estimate is ``errors / bits``. A point estimate without an
interval is not a result, so every estimate carries a Wilson score interval.
Wilson is used instead of the normal approximation because it behaves correctly
when the error count is small, which is exactly the high-SNR regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "BerEstimate",
    "BerPoint",
    "ber_estimate",
    "count_bit_errors",
    "wilson_interval",
]


@dataclass(frozen=True)
class BerEstimate:
    """A Monte Carlo bit-error estimate with a binomial confidence interval."""

    errors: int
    bits: int
    ber: float
    ci_low: float
    ci_high: float
    confidence: float

    def as_dict(self) -> dict[str, int | float]:
        """Return the estimate as a plain dictionary for JSON or CSV output."""
        return {
            "errors": self.errors,
            "bits": self.bits,
            "ber": self.ber,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class BerPoint:
    """Aggregate and per-stream BER estimates for one SNR point.

    Decision D-07 requires both. Under i.i.d. channels the two agree in
    expectation, but the per-stream counts make the distinction explicit and
    become material once spatial correlation is added.
    """

    aggregate: BerEstimate
    per_stream: tuple[BerEstimate, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the point as a nested dictionary."""
        return {
            "aggregate": self.aggregate.as_dict(),
            "per_stream": [estimate.as_dict() for estimate in self.per_stream],
        }


def _validate_counts(errors: int, bits: int) -> None:
    if not isinstance(errors, (int, np.integer)) or isinstance(errors, bool):
        raise ValueError("errors must be an integer")
    if not isinstance(bits, (int, np.integer)) or isinstance(bits, bool):
        raise ValueError("bits must be an integer")
    if bits <= 0:
        raise ValueError("bits must be positive")
    if errors < 0 or errors > bits:
        raise ValueError("errors must satisfy 0 <= errors <= bits")


def _validate_confidence(confidence: float) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a real number in (0, 1)") from exc
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return value


def count_bit_errors(transmitted: ArrayLike, estimated: ArrayLike) -> int:
    """Count differing bits between two equally shaped bit arrays.

    Parameters
    ----------
    transmitted, estimated
        Arrays of 0/1 values with identical shapes.

    Returns
    -------
    int
        Number of positions where the two arrays differ.
    """
    tx = np.asarray(transmitted)
    rx = np.asarray(estimated)

    if tx.shape != rx.shape:
        raise ValueError(f"bit arrays must have equal shape; got {tx.shape} and {rx.shape}")
    if tx.size == 0:
        raise ValueError("bit arrays must not be empty")
    if not np.all((tx == 0) | (tx == 1)) or not np.all((rx == 0) | (rx == 1)):
        raise ValueError("bit arrays must contain only 0 and 1")

    return int(np.count_nonzero(tx != rx))


def wilson_interval(
    errors: int,
    bits: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    errors
        Number of observed errors, ``0 <= errors <= bits``.
    bits
        Number of observed bits, positive.
    confidence
        Confidence level in ``(0, 1)``.

    Returns
    -------
    tuple of float
        Lower and upper bounds, clipped to ``[0, 1]``.
    """
    _validate_counts(errors, bits)
    level = _validate_confidence(confidence)

    z = NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)
    proportion = errors / bits
    denominator = 1.0 + z * z / bits
    centre = (proportion + z * z / (2.0 * bits)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / bits + z * z / (4.0 * bits * bits)
        )
        / denominator
    )

    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def ber_estimate(
    errors: int,
    bits: int,
    confidence: float = 0.95,
) -> BerEstimate:
    """Build a BER estimate with its Wilson interval."""
    low, high = wilson_interval(errors, bits, confidence=confidence)
    return BerEstimate(
        errors=int(errors),
        bits=int(bits),
        ber=errors / bits,
        ci_low=low,
        ci_high=high,
        confidence=float(confidence),
    )
