"""Channel: fading models and additive noise generation.

Implemented here:

- :mod:`wireless_phy_lab.channel.awgn` -- complex AWGN with ``sigma^2 =
  1/(Es/N0)`` calibration, seeded generation, and signal addition.

Fading models are not implemented yet.
"""

from wireless_phy_lab.channel import awgn

__all__ = ["awgn"]
