"""Tests for the closed-form reference curves, equations (8.2) and (8.4)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfc

from wireless_phy_lab.evaluation.reference import awgn_qpsk_ber, qfunc, zf_rayleigh_ber


def test_qfunc_matches_reference_anchors() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.array(
        [
            1.586552539314571e-1,
            2.275013194817922e-2,
            1.349898031630096e-3,
            3.167124183311998e-5,
        ]
    )
    np.testing.assert_allclose(qfunc(x), expected, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(qfunc(3.0), 0.5 * erfc(3.0 / np.sqrt(2.0)), rtol=1e-15)


@pytest.mark.parametrize(
    ("eb_n0", "expected"),
    [
        (0.5, 1.586552539314571e-1),   # Es/N0 = 0 dB
        (1.0, 7.864960352514258e-2),   # Eb/N0 = 0 dB
        (5.0, 7.82701e-4),             # Es/N0 = 10 dB, model V2 precision
    ],
)
def test_awgn_qpsk_ber_matches_model_anchors(eb_n0: float, expected: float) -> None:
    np.testing.assert_allclose(awgn_qpsk_ber(eb_n0), expected, rtol=1e-6)


@pytest.mark.parametrize(
    ("es_n0", "diversity_order", "expected"),
    [
        (10.0, 1, 4.35645e-2),
        (100.0, 1, 4.92623e-3),
        (10.0 ** 1.2, 2, 2.448087e-3),
        (10.0 ** 1.2, 3, 2.297337e-4),
        (10.0 ** 1.2, 4, 2.259215e-5),
    ],
)
def test_zf_rayleigh_ber_matches_model_anchors(
    es_n0: float, diversity_order: int, expected: float
) -> None:
    np.testing.assert_allclose(
        zf_rayleigh_ber(es_n0, diversity_order), expected, rtol=2e-5
    )


def test_zf_rayleigh_m1_matches_simple_formula() -> None:
    rho = np.array([1.0, 10.0, 100.0])
    expected = 0.5 * (1.0 - np.sqrt(rho / (rho + 2.0)))
    np.testing.assert_allclose(zf_rayleigh_ber(rho, 1), expected, rtol=1e-14)


def test_reference_curves_preserve_shape() -> None:
    eb_n0 = np.array([0.5, 1.0, 5.0])
    assert awgn_qpsk_ber(eb_n0).shape == (3,)
    assert zf_rayleigh_ber(np.array([1.0, 10.0]), 2).shape == (2,)


@pytest.mark.parametrize("eb_n0", [-1.0, np.nan, np.inf])
def test_awgn_qpsk_ber_rejects_invalid_eb_n0(eb_n0: float) -> None:
    with pytest.raises(ValueError):
        awgn_qpsk_ber(eb_n0)


@pytest.mark.parametrize("es_n0", [-1.0, np.nan, np.inf])
def test_zf_rayleigh_ber_rejects_invalid_es_n0(es_n0: float) -> None:
    with pytest.raises(ValueError):
        zf_rayleigh_ber(es_n0, 1)


@pytest.mark.parametrize("diversity_order", [0, -1, 1.5, True])
def test_zf_rayleigh_ber_rejects_invalid_diversity_order(
    diversity_order: object,
) -> None:
    with pytest.raises(ValueError):
        zf_rayleigh_ber(10.0, diversity_order)  # type: ignore[arg-type]
