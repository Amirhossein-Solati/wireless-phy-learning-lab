"""Evaluation: BER metrics, Monte Carlo scheduling, and reference curves.

Implemented here:

- :mod:`wireless_phy_lab.evaluation.ber` -- bit error counting, Wilson
  confidence intervals, and the :class:`BerEstimate` record.
- :mod:`wireless_phy_lab.evaluation.monte_carlo` -- seeded BER simulation for
  the QPSK MIMO baseline.
- :mod:`wireless_phy_lab.evaluation.reference` -- exact AWGN QPSK and ZF
  Rayleigh reference curves.
"""

from wireless_phy_lab.evaluation import ber, monte_carlo, reference
from wireless_phy_lab.evaluation.ber import (
    BerEstimate,
    BerPoint,
    ber_estimate,
    count_bit_errors,
    wilson_interval,
)
from wireless_phy_lab.evaluation.monte_carlo import simulate_ber_point

__all__ = [
    "BerEstimate",
    "BerPoint",
    "ber",
    "ber_estimate",
    "count_bit_errors",
    "monte_carlo",
    "reference",
    "simulate_ber_point",
    "wilson_interval",
]
