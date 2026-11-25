"""Tests for the experiment and plotting scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from experiments import ber_mimo, plot_ber_mimo


def _base_config() -> dict:
    return {
        "seed": 12345,
        "confidence": 0.95,
        "replicates": 2,
        "target_errors": 5,
        "max_bits": 4000,
        "batch_symbols": 200,
        "systems": [
            {
                "name": "2x2",
                "nr": 2,
                "nt": 2,
                "detectors": {
                    "zf": [0.0, 6.0],
                    "mmse": [0.0, 6.0],
                    "ml": [0.0],
                },
            }
        ],
        "condition_samples": 100,
        "condition_threshold": 100.0,
        "output_dir": "unused",
        "figure_dir": "unused",
    }


def _write_config(path: Path, config: dict) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def test_default_config_loads() -> None:
    config = ber_mimo._load_config(ber_mimo.DEFAULT_CONFIG)
    assert config["replicates"] >= 2
    assert config["systems"]
    assert any("zf" in system["detectors"] for system in config["systems"])


def test_point_seeds_are_unique_and_deterministic() -> None:
    first = [
        ber_mimo._point_seed(20261127, system, snr)
        for system in range(2)
        for snr in range(1001)
    ]
    second = [
        ber_mimo._point_seed(20261127, system, snr)
        for system in range(2)
        for snr in range(1001)
    ]
    assert first == second
    assert len(set(first)) == len(first)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.pop("replicates"),
        lambda config: config["systems"][0]["detectors"].update({"unknown": [0.0]}),
        lambda config: config["systems"][0]["detectors"].update({"zf": [0.0, 0.0]}),
        lambda config: config["systems"][0]["detectors"].update({"zf": []}),
        lambda config: config.update(
            {
                "systems": [
                    {"name": "x", "nr": 2, "nt": 2, "detectors": {"zf": [0.0]}},
                    {"name": "x", "nr": 4, "nt": 4, "detectors": {"zf": [0.0]}},
                ]
            }
        ),
        lambda config: config.update({"replicates": 1}),
        lambda config: config.update({"systems": []}),
    ],
)
def test_invalid_config_is_rejected(mutation, tmp_path: Path) -> None:
    config = _base_config()
    mutation(config)
    path = _write_config(tmp_path / "config.yaml", config)
    with pytest.raises(ValueError):
        ber_mimo._load_config(path)


def test_run_experiment_writes_and_plots_results(tmp_path: Path) -> None:
    config = _base_config()
    config_path = _write_config(tmp_path / "config.yaml", config)
    results_dir = tmp_path / "results"

    payload = ber_mimo.run_experiment(config, results_dir, config_path=config_path)

    assert len(payload["results"]) == 5
    assert "git_commit" in payload
    assert len(payload["channel_conditions"]) == 1

    for record in payload["results"]:
        assert record["replicates"] == 2
        assert record["target_errors"] == 5
        assert record["max_bits"] == 4000
        assert len(record["per_stream"]) == 2
        assert len(record["replicate_seeds"]) == 2
        assert len(set(record["replicate_seeds"])) == 2
        assert record["ci_low"] <= record["ber"] <= record["ci_high"]
        if record["detector"] == "zf":
            assert record["condition"] is not None
            assert record["condition"]["samples"] == 100
        else:
            assert record["condition"] is None

    json_path = results_dir / "ber_mimo.json"
    csv_path = results_dir / "ber_mimo.csv"
    config_copy = results_dir / "ber_mimo_config.yaml"
    assert json_path.exists()
    assert csv_path.exists()
    assert config_copy.exists()
    assert config_copy.read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["config"] == config
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert rows[0]["condition_median"]

    figure = plot_ber_mimo._plot_system(
        config["systems"][0], payload["results"], tmp_path / "figures"
    )
    assert figure.exists()
    assert figure.stat().st_size > 0
