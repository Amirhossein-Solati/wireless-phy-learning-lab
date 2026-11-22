"""Tests for the seeded Monte Carlo BER runner."""

from __future__ import annotations

import numpy as np
import pytest

from wireless_phy_lab.detection import mmse_detect, zf_detect
from wireless_phy_lab.evaluation.monte_carlo import simulate_ber_point
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
