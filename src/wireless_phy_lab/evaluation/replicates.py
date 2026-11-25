"""Independent-replicate BER summaries.

A single long run is not a set of independent replicates: bits from different
spatial streams share the same channel matrix and noise vector, so the
aggregate error count is overdispersed relative to a binomial. This module
runs ``R`` independent replicates with spawned seeds and reports a Student-t
interval for the mean replicate BER. Per-stream summaries use the same scheme.

The point estimate is the mean of the replicate BERs, not the pooled error
ratio. Both the pooled counts and the replicate BERs are retained for audit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

from wireless_phy_lab.evaluation.ber import BerEstimate
from wireless_phy_lab.evaluation.monte_carlo import Detector, simulate_ber_point

__all__ = [
    "BerReplicatePoint",
    "BerSummary",
    "simulate_ber_replicates",
    "summarize_replicates",
    "t_interval",
]


@dataclass(frozen=True)
class BerSummary:
    """Mean replicate BER with a Student-t confidence interval."""

    ber: float
    ci_low: float
    ci_high: float
    confidence: float
    replicates: int
    total_errors: int
    total_bits: int
    replicate_bers: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the summary as a JSON-serialisable dictionary."""
        return {
            "ber": self.ber,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "replicates": self.replicates,
            "total_errors": self.total_errors,
            "total_bits": self.total_bits,
            "replicate_bers": list(self.replicate_bers),
        }


@dataclass(frozen=True)
class BerReplicatePoint:
    """Aggregate and per-stream replicate summaries for one SNR point."""

    aggregate: BerSummary
    per_stream: tuple[BerSummary, ...]
    replicate_seeds: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the point as a JSON-serialisable dictionary."""
        return {
            "aggregate": self.aggregate.as_dict(),
            "per_stream": [summary.as_dict() for summary in self.per_stream],
            "replicate_seeds": list(self.replicate_seeds),
        }


def t_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Student-t interval for the mean of independent replicate estimates.

    Parameters
    ----------
    values
        One BER estimate per independent replicate. At least two values.
    confidence
        Confidence level in ``(0, 1)``.

    Returns
    -------
    tuple of float
        ``(mean, lower, upper)``, clipped to ``[0, 1]``.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2:
        raise ValueError("at least two replicate values are required")
    if not np.all(np.isfinite(data)) or np.any(data < 0.0) or np.any(data > 1.0):
        raise ValueError("replicate BERs must be finite and in [0, 1]")

    level = float(confidence)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    mean = float(data.mean())
    standard_error = float(data.std(ddof=1) / np.sqrt(data.size))
    critical = float(student_t.ppf(1.0 - (1.0 - level) / 2.0, df=data.size - 1))
    half_width = critical * standard_error

    return mean, max(0.0, mean - half_width), min(1.0, mean + half_width)


def summarize_replicates(
    estimates: Sequence[BerEstimate],
    confidence: float = 0.95,
) -> BerSummary:
    """Summarise one BER coordinate across independent replicates."""
    if not estimates:
        raise ValueError("at least one replicate estimate is required")

    values = np.array([estimate.ber for estimate in estimates], dtype=np.float64)
    mean, low, high = t_interval(values, confidence=confidence)
    return BerSummary(
        ber=mean,
        ci_low=low,
        ci_high=high,
        confidence=float(confidence),
        replicates=len(estimates),
        total_errors=int(sum(estimate.errors for estimate in estimates)),
        total_bits=int(sum(estimate.bits for estimate in estimates)),
        replicate_bers=tuple(float(value) for value in values),
    )


def simulate_ber_replicates(
    detector: Detector,
    *,
    nr: int,
    nt: int,
    eb_n0_db: float,
    seed: int,
    replicates: int = 20,
    target_errors: int = 200,
    max_bits: int = 2_000_000,
    batch_symbols: int = 20_000,
    confidence: float = 0.95,
) -> BerReplicatePoint:
    """Run independent replicates and return t-interval BER summaries.

    Each replicate uses a child seed spawned from ``seed``. Replicates are
    independent, so the t-interval is valid for the mean replicate BER. The
    aggregate and per-stream summaries are computed from the same replicate
    runs. Both detectors at a point share the base seed when the experiment
    passes the same value, which keeps their channel/noise streams paired.

    ``target_errors`` and ``max_bits`` apply to each replicate, not to the
    pooled run.
    """
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, (int, np.integer))
        or replicates < 2
    ):
        raise ValueError("replicates must be an integer of at least 2")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    children = np.random.SeedSequence([int(seed)]).spawn(int(replicates))
    replicate_seeds = tuple(int(child.generate_state(1)[0]) for child in children)

    points = [
        simulate_ber_point(
            detector,
            nr=nr,
            nt=nt,
            eb_n0_db=eb_n0_db,
            seed=replicate_seed,
            target_errors=target_errors,
            max_bits=max_bits,
            batch_symbols=batch_symbols,
            confidence=confidence,
        )
        for replicate_seed in replicate_seeds
    ]

    aggregate = summarize_replicates(
        [point.aggregate for point in points], confidence=confidence
    )
    per_stream = tuple(
        summarize_replicates(
            [point.per_stream[stream] for point in points],
            confidence=confidence,
        )
        for stream in range(nt)
    )

    return BerReplicatePoint(
        aggregate=aggregate,
        per_stream=per_stream,
        replicate_seeds=replicate_seeds,
    )
