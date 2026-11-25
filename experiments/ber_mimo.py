"""Run the 2x2 and 4x4 QPSK BER experiment with independent replicates.

Usage
-----
    conda activate phy-lab
    python experiments/ber_mimo.py

The configuration is read from ``experiments/configs/ber_mimo.yaml`` by
default. Results are written to ``results/ber_mimo.json`` and
``results/ber_mimo.csv``; the exact configuration is copied to
``results/ber_mimo_config.yaml``.

Statistical design
------------------
Each point runs ``replicates`` independent child seeds and reports a Student-t
interval for the mean replicate BER. This is deliberately not a single-run
binomial Wilson interval: spatial streams share the same channel and noise, so
the aggregate error count is overdispersed. Per-stream summaries use the same
replicate scheme. Both detectors at a point use the same point seed, which
keeps their channel/noise streams paired.

Condition numbers are reported from a separate fixed-size i.i.d. channel sample
with a recorded diagnostic seed, as required for ZF results.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import scipy
import yaml

from wireless_phy_lab import __version__
from wireless_phy_lab.detection import ml_detect, mmse_detect, zf_detect
from wireless_phy_lab.evaluation.diagnostics import (
    ConditionSummary,
    condition_number_summary,
)
from wireless_phy_lab.evaluation.replicates import (
    BerReplicatePoint,
    simulate_ber_replicates,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "ber_mimo.yaml"

DETECTOR_ORDER = ("zf", "mmse", "ml")
DETECTORS = {
    "zf": lambda channel, received, noise_variance: zf_detect(channel, received),
    "mmse": mmse_detect,
    "ml": lambda channel, received, noise_variance: ml_detect(channel, received),
}

CSV_FIELDS = [
    "system",
    "nr",
    "nt",
    "detector",
    "eb_n0_db",
    "seed",
    "replicates",
    "target_errors",
    "max_bits",
    "total_errors",
    "total_bits",
    "ber",
    "ci_low",
    "ci_high",
    "confidence",
    "condition_median",
    "condition_p95",
    "condition_max",
    "condition_fraction_above",
    "condition_samples",
    "condition_seed",
]


def _seed(*parts: int) -> int:
    """Return a reproducible 32-bit seed from integer parts."""
    return int(np.random.SeedSequence([int(part) for part in parts]).generate_state(1)[0])


def _point_seed(base_seed: int, system_index: int, snr_key: int) -> int:
    """Seed shared by detectors at one system/SNR point."""
    return _seed(base_seed, system_index, snr_key)


def _condition_seed(base_seed: int, system_index: int) -> int:
    """Independent seed for the condition-number diagnostic sample."""
    return _seed(base_seed, 0x434F4E44, system_index)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required = {
        "seed",
        "confidence",
        "replicates",
        "target_errors",
        "max_bits",
        "batch_symbols",
        "systems",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"config is missing keys: {sorted(missing)}")

    if not config["systems"]:
        raise ValueError("systems must not be empty")
    names = [system["name"] for system in config["systems"]]
    if len(set(names)) != len(names):
        raise ValueError("system names must be unique")

    for system in config["systems"]:
        detector_grids = system.get("detectors")
        if not detector_grids:
            raise ValueError(f"system {system.get('name')} has no detectors")
        unknown = set(detector_grids) - set(DETECTORS)
        if unknown:
            raise ValueError(f"unknown detectors: {sorted(unknown)}")
        for detector_name, grid in detector_grids.items():
            if not grid:
                raise ValueError(
                    f"system {system['name']} detector {detector_name} has an empty grid"
                )
            if len(set(grid)) != len(grid):
                raise ValueError(
                    f"system {system['name']} detector {detector_name} has duplicate SNR values"
                )

    if int(config["replicates"]) < 2:
        raise ValueError("replicates must be at least 2")
    if int(config["target_errors"]) < 1:
        raise ValueError("target_errors must be positive")
    if int(config["max_bits"]) < 1:
        raise ValueError("max_bits must be positive")
    if int(config["batch_symbols"]) < 1:
        raise ValueError("batch_symbols must be positive")

    return config


def _record(
    system: dict,
    detector: str,
    eb_n0_db: float,
    seed: int,
    point: BerReplicatePoint,
    target_errors: int,
    max_bits: int,
    condition: ConditionSummary | None,
) -> dict:
    summary = point.aggregate
    return {
        "system": system["name"],
        "nr": int(system["nr"]),
        "nt": int(system["nt"]),
        "detector": detector,
        "eb_n0_db": float(eb_n0_db),
        "seed": seed,
        "replicates": summary.replicates,
        "target_errors": int(target_errors),
        "max_bits": int(max_bits),
        "total_errors": summary.total_errors,
        "total_bits": summary.total_bits,
        "ber": summary.ber,
        "ci_low": summary.ci_low,
        "ci_high": summary.ci_high,
        "confidence": summary.confidence,
        "per_stream": [entry.as_dict() for entry in point.per_stream],
        "replicate_seeds": list(point.replicate_seeds),
        "condition": condition.as_dict() if condition is not None else None,
    }


def _git_state() -> tuple[str | None, bool | None]:
    """Return the current commit and whether the code paths are dirty."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(REPO_ROOT),
                    "status",
                    "--porcelain",
                    "--",
                    "src",
                    "tests",
                    "experiments",
                    "pyproject.toml",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def run_experiment(
    config: dict,
    results_dir: Path,
    config_path: Path | None = None,
) -> dict:
    """Run the configured experiment and write JSON, CSV and config copy."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(config["seed"])
    records: list[dict] = []
    condition_records: list[dict] = []

    for system_index, system in enumerate(config["systems"]):
        condition = condition_number_summary(
            nr=int(system["nr"]),
            nt=int(system["nt"]),
            seed=_condition_seed(base_seed, system_index),
            samples=int(config.get("condition_samples", 10_000)),
            threshold=float(config.get("condition_threshold", 100.0)),
        )
        condition_records.append({"system": system["name"], **condition.as_dict()})

        detector_grids = system["detectors"]
        for detector_name in DETECTOR_ORDER:
            if detector_name not in detector_grids:
                continue
            for eb_n0_db in detector_grids[detector_name]:
                seed = _point_seed(base_seed, system_index, int(round(10.0 * eb_n0_db)))
                point = simulate_ber_replicates(
                    DETECTORS[detector_name],
                    nr=int(system["nr"]),
                    nt=int(system["nt"]),
                    eb_n0_db=float(eb_n0_db),
                    seed=seed,
                    replicates=int(config["replicates"]),
                    target_errors=int(config["target_errors"]),
                    max_bits=int(config["max_bits"]),
                    batch_symbols=int(config["batch_symbols"]),
                    confidence=float(config["confidence"]),
                )
                summary = point.aggregate
                record = _record(
                    system,
                    detector_name,
                    eb_n0_db,
                    seed,
                    point,
                    int(config["target_errors"]),
                    int(config["max_bits"]),
                    condition if detector_name == "zf" else None,
                )
                records.append(record)
                print(
                    f"{system['name']:>3} {detector_name:>4} {eb_n0_db:5.1f} dB  "
                    f"BER={summary.ber:.4e}  "
                    f"CI=[{summary.ci_low:.3e}, {summary.ci_high:.3e}]  "
                    f"replicates={summary.replicates}  "
                    f"errors/bits={summary.total_errors}/{summary.total_bits}  "
                    f"seed={seed}",
                    flush=True,
                )

    config_copy = results_dir / "ber_mimo_config.yaml"
    if config_path is not None:
        shutil.copyfile(config_path, config_copy)
    else:
        with config_copy.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    commit, dirty = _git_state()
    payload = {
        "config": config,
        "config_path": str(config_path) if config_path is not None else None,
        "seed_policy": (
            "point seed = SeedSequence([base_seed, system_index, snr_index]); "
            "both detectors share the point seed; replicate seeds are spawned "
            "children of the point seed"
        ),
        "git_commit": commit,
        "git_dirty": dirty,
        "environment": {
            "package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "channel_conditions": condition_records,
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
        for record in records:
            condition = record["condition"] or {}
            row = {field: record.get(field) for field in CSV_FIELDS}
            row.update(
                {
                    "condition_median": condition.get("median"),
                    "condition_p95": condition.get("p95"),
                    "condition_max": condition.get("maximum"),
                    "condition_fraction_above": condition.get("fraction_above"),
                    "condition_samples": condition.get("samples"),
                    "condition_seed": condition.get("seed"),
                }
            )
            writer.writerow(row)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {config_copy}")
    return payload


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
    run_experiment(config, results_dir, config_path=config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
