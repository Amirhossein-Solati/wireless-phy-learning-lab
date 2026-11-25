"""Tests for the seeded Monte Carlo BER runner."""

from __future__ import annotations

import numpy as np
import pytest

from wireless_phy_lab.detection import mmse_detect, zf_detect
from wireless_phy_lab.evaluation.monte_carlo import simulate_ber_point
from wireless_phy_lab.evaluation.reference import zf_rayleigh_ber
from wireless_phy_lab.evaluation.replicates import simulate_ber_replicates
from wireless_phy_lab.modulation import qpsk

SEED = 20261127


def _zf(channel: np.ndarray, received: np.ndarray, noise_variance: float) -> np.ndarray:
    return zf_detect(channel, received)


def _zero_detector(
    channel: np.ndarray, received: np.ndarray, noise_variance: float
) -> np.ndarray:
    return np.zeros((channel.shape[0], channel.shape[-1]), dtype=np.complex128)


def test_simulation_is_reproducible_for_a_fixed_seed() -> None:
    kwargs = {
        "nr": 2,
        "nt": 2,
        "eb_n0_db": 6.0,
        "seed": SEED,
        "target_errors": 20,
        "max_bits": 20_000,
        "batch_symbols": 1_000,
        "confidence": 0.95,
    }
    first = simulate_ber_point(_zf, **kwargs)
    second = simulate_ber_point(_zf, **kwargs)
    other = simulate_ber_point(_zf, **{**kwargs, "seed": SEED + 1})

    assert first == second
    assert (other.aggregate.errors, other.aggregate.bits) != (
        first.aggregate.errors,
        first.aggregate.bits,
    )


def test_simulation_result_is_internally_consistent() -> None:
    estimate = simulate_ber_point(
        _zf,
        nr=2,
        nt=2,
        eb_n0_db=6.0,
        seed=SEED,
        target_errors=20,
        max_bits=20_000,
        batch_symbols=1_000,
        confidence=0.95,
    )

    assert estimate.aggregate.errors >= 20
    assert estimate.aggregate.bits <= 20_000
    assert estimate.aggregate.bits % (2 * qpsk.BITS_PER_SYMBOL) == 0
    assert estimate.aggregate.ci_low <= estimate.aggregate.ber <= estimate.aggregate.ci_high
    assert estimate.aggregate.confidence == 0.95
    assert len(estimate.per_stream) == 2
    assert sum(item.errors for item in estimate.per_stream) == estimate.aggregate.errors
    assert sum(item.bits for item in estimate.per_stream) == estimate.aggregate.bits


def test_zero_detector_gives_half_errors() -> None:
    estimate = simulate_ber_point(
        _zero_detector,
        nr=2,
        nt=2,
        eb_n0_db=0.0,
        seed=SEED,
        target_errors=100,
        max_bits=20_000,
        batch_symbols=1_000,
    )
    assert 0.4 < estimate.aggregate.ber < 0.6


def test_max_bits_stops_the_simulation() -> None:
    estimate = simulate_ber_point(
        _zero_detector,
        nr=2,
        nt=2,
        eb_n0_db=0.0,
        seed=SEED,
        target_errors=10_000_000,
        max_bits=4_000,
        batch_symbols=1_000,
    )
    assert estimate.aggregate.bits == 4_000
    assert estimate.aggregate.errors < 10_000_000


def test_mmse_runner_accepts_the_detector_signature() -> None:
    estimate = simulate_ber_point(
        mmse_detect,
        nr=2,
        nt=2,
        eb_n0_db=6.0,
        seed=SEED,
        target_errors=10,
        max_bits=10_000,
        batch_symbols=1_000,
    )
    assert estimate.aggregate.errors >= 10
    assert 0.0 <= estimate.aggregate.ber <= 1.0


def test_detector_must_return_the_expected_shape() -> None:
    def wrong_shape(
        channel: np.ndarray, received: np.ndarray, noise_variance: float
    ) -> np.ndarray:
        return np.zeros((channel.shape[0], 1), dtype=np.complex128)

    with pytest.raises(ValueError, match="detector returned"):
        simulate_ber_point(
            wrong_shape,
            nr=2,
            nt=2,
            eb_n0_db=6.0,
            seed=SEED,
            target_errors=1,
            max_bits=100,
            batch_symbols=10,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"nr": 0},
        {"nt": 0},
        {"target_errors": 0},
        {"max_bits": 3},
        {"batch_symbols": 0},
        {"seed": -1},
        {"eb_n0_db": np.nan},
    ],
)
def test_invalid_configuration_is_rejected(overrides: dict[str, object]) -> None:
    config: dict[str, object] = {
        "nr": 2,
        "nt": 2,
        "eb_n0_db": 6.0,
        "seed": SEED,
        "target_errors": 10,
        "max_bits": 10_000,
        "batch_symbols": 1_000,
    }
    config.update(overrides)
    with pytest.raises(ValueError):
        simulate_ber_point(_zf, **config)  # type: ignore[arg-type]


def test_runner_passes_the_calibrated_complex_noise_variance() -> None:
    """The detector must receive sigma^2 = 1/(Es/N0), not an SNR or Eb/N0."""
    captured: dict[str, float] = {}

    def recording_detector(
        channel: np.ndarray, received: np.ndarray, noise_variance: float
    ) -> np.ndarray:
        captured["noise_variance"] = noise_variance
        return zf_detect(channel, received)

    eb_n0_db = 6.0
    simulate_ber_point(
        recording_detector,
        nr=2,
        nt=2,
        eb_n0_db=eb_n0_db,
        seed=SEED,
        target_errors=1,
        max_bits=4,
        batch_symbols=1,
    )

    expected = 1.0 / float(qpsk.es_n0_from_eb_n0(10.0 ** (eb_n0_db / 10.0)))
    assert captured["noise_variance"] == pytest.approx(expected, rel=1e-15)


def test_zf_end_to_end_matches_closed_form() -> None:
    """Full chain 2x2 ZF BER lies inside the replicate t-interval of (8.4)."""
    es_n0 = float(qpsk.es_n0_from_eb_n0(10.0 ** (10.0 / 10.0)))
    reference = float(zf_rayleigh_ber(es_n0, 1))
    point = simulate_ber_replicates(
        _zf,
        nr=2,
        nt=2,
        eb_n0_db=10.0,
        seed=SEED + 20,
        replicates=20,
        target_errors=100,
        max_bits=200_000,
        batch_symbols=5_000,
    )
    assert point.aggregate.ci_low <= reference <= point.aggregate.ci_high


def test_zf_4x2_end_to_end_matches_m3_closed_form() -> None:
    """The rectangular m=3 validation case of T-09 is exercised end to end."""
    es_n0 = float(qpsk.es_n0_from_eb_n0(10.0 ** (12.0 / 10.0)))
    reference = float(zf_rayleigh_ber(es_n0, 3))
    point = simulate_ber_replicates(
        _zf,
        nr=4,
        nt=2,
        eb_n0_db=12.0,
        seed=SEED + 21,
        replicates=20,
        target_errors=50,
        max_bits=500_000,
        batch_symbols=5_000,
    )
    assert point.aggregate.ci_low <= reference <= point.aggregate.ci_high


def test_mmse_end_to_end_is_better_than_zf_at_low_snr() -> None:
    kwargs = {
        "nr": 2,
        "nt": 2,
        "eb_n0_db": 0.0,
        "seed": SEED + 22,
        "replicates": 10,
        "target_errors": 100,
        "max_bits": 100_000,
        "batch_symbols": 5_000,
    }
    zf_point = simulate_ber_replicates(_zf, **kwargs)
    mmse_point = simulate_ber_replicates(mmse_detect, **kwargs)
    assert mmse_point.aggregate.ber < zf_point.aggregate.ber
