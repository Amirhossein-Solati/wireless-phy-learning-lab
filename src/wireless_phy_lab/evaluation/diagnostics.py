"""Channel condition-number diagnostics.

The model requires a condition-number report alongside ZF results because ZF
solves the normal equations and squares the condition number of ``H``. The
diagnostic is computed on a fixed-size i.i.d. channel sample drawn with its own
recorded seed, independent of the BER simulation. It therefore describes the
channel distribution at that system size, not the exact matrices used for a
particular BER point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wireless_phy_lab.channel.rayleigh import generate_rayleigh
from wireless_phy_lab.detection.linear import condition_number

__all__ = ["ConditionSummary", "condition_number_summary"]


@dataclass(frozen=True)
class ConditionSummary:
    """Summary of a condition-number sample."""

    median: float
    p95: float
    maximum: float
    fraction_above: float
    threshold: float
    samples: int
    seed: int

    def as_dict(self) -> dict[str, int | float]:
        """Return the summary as a JSON-serialisable dictionary."""
        return {
            "median": self.median,
            "p95": self.p95,
            "maximum": self.maximum,
            "fraction_above": self.fraction_above,
            "threshold": self.threshold,
            "samples": self.samples,
            "seed": self.seed,
        }


def condition_number_summary(
    nr: int,
    nt: int,
    seed: int,
    samples: int = 10_000,
    threshold: float = 100.0,
) -> ConditionSummary:
    """Summarise the 2-norm condition number of i.i.d. Rayleigh channels.

    Parameters
    ----------
    nr, nt
        Receive and transmit antenna counts.
    seed
        Seed for the diagnostic channel sample. This is independent of the BER
        simulation seed and is recorded with the result.
    samples
        Number of channel matrices in the diagnostic sample.
    threshold
        Condition number above which the fraction is reported. A value of 100
        means the normal equations lose about two decimal digits relative to
        ``H``.
    """
    if (
        isinstance(samples, bool)
        or not isinstance(samples, (int, np.integer))
        or samples < 1
    ):
        raise ValueError("samples must be a positive integer")
    if not np.isfinite(threshold) or threshold <= 1.0:
        raise ValueError("threshold must be finite and greater than one")

    rng = np.random.default_rng(int(seed))
    channel = generate_rayleigh(nr, nt, (int(samples),), rng=rng)
    values = np.asarray(condition_number(channel), dtype=np.float64)

    return ConditionSummary(
        median=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
        maximum=float(np.max(values)),
        fraction_above=float(np.mean(values > threshold)),
        threshold=float(threshold),
        samples=int(samples),
        seed=int(seed),
    )
