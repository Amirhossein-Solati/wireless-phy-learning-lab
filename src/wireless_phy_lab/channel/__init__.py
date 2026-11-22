"""Channel: fading models and additive noise generation.

Implemented here:

- :mod:`wireless_phy_lab.channel.awgn` -- complex AWGN with ``sigma^2 =
  1/(Es/N0)`` calibration, seeded generation, and signal addition.
- :mod:`wireless_phy_lab.channel.rayleigh` -- flat i.i.d. Rayleigh fading with
  ``h_ij ~ CN(0, 1)`` and optional independent batches of channel matrices.

Correlated and time-varying fading models are not implemented yet.
"""

from wireless_phy_lab.channel import awgn, rayleigh

__all__ = ["awgn", "rayleigh"]
