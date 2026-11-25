"""Evaluation: BER metrics, Monte Carlo scheduling, and reference curves.

Implemented here:

- :mod:`wireless_phy_lab.evaluation.ber` -- bit error counting, Wilson
  confidence intervals, and the :class:`BerEstimate` record.
- :mod:`wireless_phy_lab.evaluation.monte_carlo` -- seeded BER simulation for
  the QPSK MIMO baseline.
- :mod:`wireless_phy_lab.evaluation.replicates` -- independent-replicate BER
  summaries with Student-t intervals.
- :mod:`wireless_phy_lab.evaluation.diagnostics` -- condition-number summaries
  for ZF reporting.
- :mod:`wireless_phy_lab.evaluation.reference` -- exact AWGN QPSK and ZF
  Rayleigh reference curves.
"""

from wireless_phy_lab.evaluation import (
    ber,
    diagnostics,
    monte_carlo,
    reference,
    replicates,
)
from wireless_phy_lab.evaluation.ber import (
    BerEstimate,
    BerPoint,
    ber_estimate,
    count_bit_errors,
    wilson_interval,
)
from wireless_phy_lab.evaluation.diagnostics import (
    ConditionSummary,
    condition_number_summary,
)
from wireless_phy_lab.evaluation.monte_carlo import simulate_ber_point
from wireless_phy_lab.evaluation.replicates import (
    BerReplicatePoint,
    BerSummary,
    simulate_ber_replicates,
    summarize_replicates,
    t_interval,
)

__all__ = [
    "BerEstimate",
    "BerPoint",
    "BerReplicatePoint",
    "BerSummary",
    "ConditionSummary",
    "ber",
    "ber_estimate",
    "condition_number_summary",
    "count_bit_errors",
    "diagnostics",
    "monte_carlo",
    "reference",
    "replicates",
    "simulate_ber_point",
    "simulate_ber_replicates",
    "summarize_replicates",
    "t_interval",
    "wilson_interval",
]
