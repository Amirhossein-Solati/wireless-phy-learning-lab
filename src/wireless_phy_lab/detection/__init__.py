"""Detection: MIMO detectors (ZF, MMSE, ML, and later learned detectors).

Implemented here:

- :mod:`wireless_phy_lab.detection.linear` -- zero forcing and biased MMSE
  linear detectors with batched channel inputs.

Maximum likelihood, successive cancellation, and learned detectors are not
implemented yet.
"""

from wireless_phy_lab.detection import linear
from wireless_phy_lab.detection.linear import condition_number, mmse_detect, zf_detect

__all__ = ["condition_number", "linear", "mmse_detect", "zf_detect"]
