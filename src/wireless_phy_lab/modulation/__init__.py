"""Modulation: constellation definition, bit mapping, and demapping.

Implemented here:

- :mod:`wireless_phy_lab.modulation.qpsk` -- Gray-mapped QPSK with unit
  average symbol energy. Mapping, hard-decision demapping, and the single
  ``Es = 2 Eb`` conversion.

Other constellations are not implemented yet.
"""

from wireless_phy_lab.modulation import qpsk

__all__ = ["qpsk"]
