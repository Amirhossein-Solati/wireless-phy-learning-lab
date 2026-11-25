# BER experiment documentation

Version: 0.1.0 (2026-11-25).

This document describes the v0.1.0 BER experiment, its statistical design, and how to reproduce
the committed results. The mathematical model is in
[`mathematical_model.md`](mathematical_model.md).

## 1. Scope

The experiment simulates uncoded Gray QPSK over narrowband, i.i.d. `CN(0,1)` Rayleigh fading with
perfect CSI and known noise variance. It covers:

- 2x2 and 4x4 spatial multiplexing,
- ZF, biased MMSE, and exhaustive ML detection,
- per-stream `Eb/N0` from 0 to 20 dB for ZF/MMSE,
- ML only over the range where the error target is reachable: 2x2 to 12 dB, 4x4 to 8 dB.

The 4x2 case is exercised end to end in the test suite against the exact `m = 3` ZF closed form; it
is not part of the committed figure set.

## 2. Estimator and confidence intervals

A single run can produce many correlated bit errors: spatial streams share the same channel matrix
and noise vector, and bits within a QPSK symbol can be dependent under fading and detection. A
single-run binomial Wilson interval on the aggregate error count is therefore too narrow.

The experiment instead runs `R = 20` independent replicates per point. Each replicate has its own
child seed and its own error count. The point estimate is the mean replicate BER,

```
Pb_hat = (1/R) sum_{r=1}^R Pb_r,
```

and the interval is a 95% Student-t interval on that mean with `R - 1` degrees of freedom. This is
valid because the replicates are independent. The pooled error and bit counts are also stored.

Per-stream summaries use the same replicate scheme. Both aggregate and per-stream results are
reported, as required by D-07. The figures show the aggregate.

Each replicate stops when it reaches 200 errors or a 2,000,000-bit safety limit. The error target is
a minimum, not an exact count: the final batch can overshoot it. The reported interval uses the
actual replicate BERs, not an assumed fixed bit count.

## 3. Seeds and pairing

The base seed is `20261127`. For a point,

```
point_seed = SeedSequence([base_seed, system_index, snr_key]).generate_state(1)[0]
```

where `snr_key = round(10 * Eb/N0_dB)`. All detectors at that system and SNR use the same point
seed, so their channel and noise streams are paired. The 20 replicate seeds are spawned children
of the point seed and are stored in the JSON.

The seed policy avoids the collision-prone `base + 1000 * index` scheme. The seed for a point does
not depend on which other detectors or SNR points are present.

## 4. Condition-number diagnostics

ZF solves the normal equations, so its numerical error grows with the square of the condition
number of `H`. The experiment reports a diagnostic sample of 10,000 i.i.d. channel matrices per
system, generated with an independent recorded seed. For each system it stores:

- median condition number,
- 95th percentile,
- maximum,
- fraction above 100,
- the threshold, sample count, and seed.

The diagnostic sample is representative of the channel distribution, not the exact matrices used
in the BER simulation. The JSON repeats the system-level condition summary on every ZF record and
also stores it under `channel_conditions`.

## 5. Configuration

[`../experiments/configs/ber_mimo.yaml`](../experiments/configs/ber_mimo.yaml) contains:

| Key | Value |
|---|---|
| `seed` | `20261127` |
| `confidence` | `0.95` |
| `replicates` | 20 |
| `target_errors` | 200 per replicate |
| `max_bits` | 2,000,000 per replicate |
| `batch_symbols` | 20,000 channel uses |
| `condition_samples` | 10,000 |
| `condition_threshold` | 100 |
| ZF/MMSE grid | 0 to 20 dB, 2 dB steps |
| ML grid, 2x2 | 0 to 12 dB, 2 dB steps |
| ML grid, 4x4 | 0 to 8 dB, 2 dB steps |

The exact configuration used is copied byte-for-byte to
[`../results/ber_mimo_config.yaml`](../results/ber_mimo_config.yaml).

## 6. Output schema

`results/ber_mimo.json` contains:

- `config`: the configuration dictionary,
- `config_path`, `seed_policy`, `git_commit`, `git_dirty` (dirty tracks code paths, not results),
- `environment`: package version, Python, NumPy, SciPy, and platform,
- `channel_conditions`: one condition summary per system,
- `results`: one record per system, detector, and SNR point.

Each result record contains the aggregate mean BER, interval, replicate count, pooled errors and
bits, per-stream summaries, replicate seeds, target errors, max bits, and the condition summary for
ZF records. Each summary contains the replicate BERs, so the t-interval can be recomputed.

`results/ber_mimo.csv` is a flat aggregate table. Condition columns are populated for ZF rows and
blank for the other detectors.

## 7. Reference curves

The ZF dashed line is equation (8.4) with `m = Nr - Nt + 1` and `rho = Es/N0`. For square arrays
`m = 1`. The AWGN QPSK reference, equation (8.2), is implemented and tested but is not shown because
it lies far below the fading curves on this axis.

The MMSE Gaussian approximation is deliberately not plotted or used as a validation target.

## 8. Committed results

The committed run contains 56 records and 182,960,000 simulated bits. The ZF closed form lies
inside the 95% interval for 20 of 22 ZF points. The two marginal exceptions are 4x4 at 4 dB
(`7.7137e-2` versus `[7.7138e-2, 7.7983e-2]`) and 4x4 at 16 dB (`6.1638e-3` versus
`[6.1701e-3, 6.4430e-3]`). Both are within about 0.1% of the interval edge; this is consistent with
95% marginal coverage over 22 points. No systematic offset is visible.

The 2x2 and 4x4 ZF intervals overlap at all 11 shared SNR points. MMSE is below ZF at all 22 shared
points, and ML is below MMSE on every measured ML point. No diversity-order slope is fitted.

## 9. Reproduction

```bash
conda activate phy-lab
python experiments/ber_mimo.py
python experiments/plot_ber_mimo.py
```

The full run takes about three minutes in the reference environment. The second command only reads
the JSON and rewrites the PNG figures.

The simulation script can be pointed at another configuration:

```bash
python experiments/ber_mimo.py --config path/to/config.yaml
```

`run_experiment(config, results_dir, config_path)` is also importable for tests; the test suite uses
a tiny configuration and a temporary directory, so CI exercises the writer and plot path without
touching the committed results.

## 10. Limitations

- ML is not measured above 12 dB for 2x2 or above 8 dB for 4x4 because the target error count is
  not reached within the bit budget.
- No diversity-order or outage analysis.
- The condition summary is a separate 10,000-matrix diagnostic sample, not a per-channel-use
  record.
- The experiment uses independent replicates, not the per-batch cluster bootstrap; both are valid,
  but only the replicate design is implemented.
- All results are simulation, not hardware measurements.
