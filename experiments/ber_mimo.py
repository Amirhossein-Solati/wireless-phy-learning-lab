"""Run the 2x2 and 4x4 QPSK BER experiment.

Usage
-----
    conda activate phy-lab
    python experiments/ber_mimo.py

The configuration is read from ``experiments/configs/ber_mimo.yaml`` by
default. Results are written to ``results/ber_mimo.json`` and
``results/ber_mimo.csv``; the exact configuration is copied to
``results/ber_mimo_config.yaml``.

Seed policy: the seed for a point is
``base_seed + 1000 * system_index + snr_index``. Both detectors at that point
use the same seed, so their channel and noise realisations are paired.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import yaml

from wireless_phy_lab.detection import mmse_detect, zf_detect
from wireless_phy_lab.evaluation.ber import BerPoint
from wireless_phy_lab.evaluation.monte_carlo import simulate_ber_point

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "ber_mimo.yaml"

DETECTORS = {
    "zf": lambda channel, received, noise_variance: zf_detect(channel, received),
    "mmse": mmse_detect,
}

CSV_FIELDS = [
    "system",
    "nr",
    "nt",
    "detector",
    "eb_n0_db",
    "seed",
    "errors",
    "bits",
    "ber",
    "ci_low",
    "ci_high",
    "confidence",
]


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required = {
        "seed",
        "confidence",
        "target_errors",
        "max_bits",
        "batch_symbols",
        "eb_n0_db",
        "systems",
        "detectors",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"config is missing keys: {sorted(missing)}")

    unknown = set(config["detectors"]) - set(DETECTORS)
    if unknown:
        raise ValueError(f"unknown detectors: {sorted(unknown)}")

    return config


def _record(
    system: dict,
    detector: str,
    eb_n0_db: float,
    seed: int,
    point: BerPoint,
) -> dict:
    estimate = point.aggregate
    return {
        "system": system["name"],
        "nr": int(system["nr"]),
        "nt": int(system["nt"]),
        "detector": detector,
        "eb_n0_db": float(eb_n0_db),
        "seed": seed,
        "errors": estimate.errors,
        "bits": estimate.bits,
        "ber": estimate.ber,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "confidence": estimate.confidence,
        "per_stream": [entry.as_dict() for entry in point.per_stream],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML configuration (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = _load_config(config_path)

    results_dir = REPO_ROOT / config.get("output_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(config["seed"])
    records: list[dict] = []

    for system_index, system in enumerate(config["systems"]):
        for snr_index, eb_n0_db in enumerate(config["eb_n0_db"]):
            seed = base_seed + 1000 * system_index + snr_index

            for detector_name in config["detectors"]:
                point = simulate_ber_point(
                    DETECTORS[detector_name],
                    nr=int(system["nr"]),
                    nt=int(system["nt"]),
                    eb_n0_db=float(eb_n0_db),
                    seed=seed,
                    target_errors=int(config["target_errors"]),
                    max_bits=int(config["max_bits"]),
                    batch_symbols=int(config["batch_symbols"]),
                    confidence=float(config["confidence"]),
                )
                estimate = point.aggregate
                record = _record(system, detector_name, eb_n0_db, seed, point)
                records.append(record)
                print(
                    f"{system['name']:>3} {detector_name:>4} {eb_n0_db:5.1f} dB  "
                    f"BER={estimate.ber:.4e}  errors={estimate.errors}/"
                    f"{estimate.bits}  CI=[{estimate.ci_low:.3e}, "
                    f"{estimate.ci_high:.3e}]  seed={seed}",
                    flush=True,
                )

    config_copy = results_dir / "ber_mimo_config.yaml"
    shutil.copyfile(config_path, config_copy)

    try:
        config_display = str(config_path.relative_to(REPO_ROOT))
    except ValueError:
        config_display = str(config_path)

    payload = {
        "config": config,
        "config_path": config_display,
        "seed_policy": (
            "base_seed + 1000 * system_index + snr_index; both detectors at a "
            "point share the seed"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "results": records,
    }

    json_path = results_dir / "ber_mimo.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    csv_path = results_dir / "ber_mimo.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: record[field] for field in CSV_FIELDS} for record in records)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {config_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
