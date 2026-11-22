"""Plot the saved 2x2 and 4x4 BER experiment.

Usage
-----
    conda activate phy-lab
    python experiments/plot_ber_mimo.py

Reads ``results/ber_mimo.json`` and writes one figure per system to
``results/figures/``. The solid lines are Monte Carlo estimates with 95% Wilson
intervals; the dashed line is the exact ZF Rayleigh curve, equation (8.4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from wireless_phy_lab.evaluation.reference import zf_rayleigh_ber
from wireless_phy_lab.modulation import qpsk

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results" / "ber_mimo.json"

STYLES = {
    "zf": {"fmt": "o-", "label": "ZF simulation"},
    "mmse": {"fmt": "s-", "label": "MMSE simulation"},
}


def _rows_for(
    results: list[dict], system_name: str, detector: str
) -> list[dict]:
    rows = [
        row
        for row in results
        if row["system"] == system_name and row["detector"] == detector
    ]
    return sorted(rows, key=lambda row: row["eb_n0_db"])


def _plot_system(system: dict, results: list[dict], figure_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for detector, style in STYLES.items():
        rows = _rows_for(results, system["name"], detector)
        if not rows:
            continue

        snr = np.array([row["eb_n0_db"] for row in rows])
        ber = np.array([row["ber"] for row in rows])
        lower = ber - np.array([row["ci_low"] for row in rows])
        upper = np.array([row["ci_high"] for row in rows]) - ber

        ax.errorbar(
            snr,
            ber,
            yerr=np.vstack([lower, upper]),
            fmt=style["fmt"],
            capsize=3,
            linewidth=1.2,
            markersize=5,
            label=style["label"],
        )

    all_rows = [row for row in results if row["system"] == system["name"]]
    snr_line = np.linspace(
        min(row["eb_n0_db"] for row in all_rows),
        max(row["eb_n0_db"] for row in all_rows),
        400,
    )
    es_n0 = qpsk.es_n0_from_eb_n0(10.0 ** (snr_line / 10.0))
    diversity_order = int(system["nr"]) - int(system["nt"]) + 1
    ax.semilogy(
        snr_line,
        zf_rayleigh_ber(es_n0, diversity_order),
        "k--",
        linewidth=1.2,
        label=f"ZF closed form (m={diversity_order})",
    )

    low = min(row["ci_low"] for row in all_rows)
    high = max(row["ci_high"] for row in all_rows)
    ax.set_ylim(max(1e-5, low / 2.0), min(0.6, high * 2.0))

    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit error rate")
    ax.set_title(f"Uncoded QPSK, i.i.d. Rayleigh, {system['name']} MIMO")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.text(
        0.03,
        0.05,
        "Error bars: 95% Wilson CI\nMMSE is biased",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()

    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"ber_mimo_{system['name']}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"results JSON (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="output directory; defaults to the config figure_dir",
    )
    args = parser.parse_args()

    results_path = args.results
    if not results_path.is_absolute():
        results_path = REPO_ROOT / results_path

    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    config = payload["config"]
    results = payload["results"]

    if args.figure_dir is None:
        figure_dir = REPO_ROOT / config.get("figure_dir", "results/figures")
    elif args.figure_dir.is_absolute():
        figure_dir = args.figure_dir
    else:
        figure_dir = REPO_ROOT / args.figure_dir

    written = []
    for system in config["systems"]:
        written.append(_plot_system(system, results, figure_dir))

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
