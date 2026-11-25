"""Detection: MIMO detectors (ZF, MMSE, ML, and later learned detectors).

Implemented here:

- :mod:`wireless_phy_lab.detection.linear` -- zero forcing and biased MMSE
  linear detectors with batched channel inputs.
- :mod:`wireless_phy_lab.detection.ml` -- exhaustive maximum-likelihood
  detection for QPSK.

Successive cancellation and learned detectors are not implemented yet.
"""

from wireless_phy_lab.detection import linear, ml
from wireless_phy_lab.detection.linear import condition_number, mmse_detect, zf_detect
from wireless_phy_lab.detection.ml import ml_detect

__all__ = [
    "condition_number",
    "linear",
    "ml",
    "ml_detect",
    "mmse_detect",
    "zf_detect",
]
