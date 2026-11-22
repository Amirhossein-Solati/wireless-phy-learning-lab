"""Smoke tests: packaging is correct and the domain subpackages import.

These tests deliberately contain no signal-processing logic. They exist so that
CI fails loudly if the package layout or install configuration breaks.
"""

import wireless_phy_lab


def test_version_is_declared() -> None:
    assert isinstance(wireless_phy_lab.__version__, str)
    assert wireless_phy_lab.__version__


def test_subpackages_are_importable() -> None:
    from wireless_phy_lab import channel, detection, evaluation, modulation

    for module in (modulation, channel, detection, evaluation):
        assert module.__name__.startswith("wireless_phy_lab.")
