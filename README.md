# Wireless PHY Learning Lab

A research-oriented simulation framework for the wireless physical layer, focused on
**narrowband spatial-multiplexing MIMO detection** with classical linear receivers and an
exhaustive maximum-likelihood reference.

> **Status: v0.1.0 research baseline.** The QPSK, Rayleigh, AWGN, ZF, MMSE, ML, and Monte Carlo
> BER components are implemented and tested. The committed results are simulations, not hardware
> measurements. ML results are reported only over the SNR range where the error target was
> reached.

- Mathematical model: [`docs/mathematical_model.md`](docs/mathematical_model.md)
- Experiment documentation: [`docs/experiments.md`](docs/experiments.md)
- Raw results: [`results/ber_mimo.json`](results/ber_mimo.json), [`results/ber_mimo.csv`](results/ber_mimo.csv)
- Figures: [`results/figures/`](results/figures/)

---

## Project objectives

1. Build a **validated classical baseline** for narrowband spatial-multiplexing MIMO. ZF is
   checked against its exact Rayleigh closed form; MMSE is checked against the explicit filter
   definition and the expected ordering against ZF and ML.
2. Provide an **exhaustive ML reference** for the MVP antenna sizes, so that linear detectors can
   be judged against the optimal detector in the same simulator.
3. Make the BER results **reproducible and statistically defensible**: recorded seeds, independent
   replicates, confidence intervals, per-stream and aggregate reporting, and stored raw counts.
4. Keep the path open to model-driven and learned detectors, which must be benchmarked against the
   ML reference before any performance claim is made.

**Non-goals for v0.1.0:** coding, LLR generation, channel estimation, OFDM, correlated or
time-varying channels, hardware measurements, and learned detectors. Diversity-order fitting is
not part of this release.

---

## System model

The narrowband model is

```
y = H x + n
```

with `Nt` transmit and `Nr` receive antennas, `x` the spatial-multiplexed transmit vector, `H` the
channel matrix, and `n` additive noise.

| Element | Choice |
|---|---|
| Modulation | Gray-mapped QPSK, unit average symbol energy |
| Antennas | `2x2` and `4x4`; `4x2` used in detector and BER validation |
| Channel | i.i.d. `CN(0,1)` Rayleigh, independent per channel use |
| Noise | circularly symmetric complex AWGN |
| CSI | perfect CSI plus known noise variance |
| Coding | none; uncoded BER |
| Detectors | ZF, biased MMSE, exhaustive ML |

After normalisation, `Es = 1` per stream and the complex noise variance is

```
sigma^2 = 1 / (Es/N0).
```

The plot axis is per-stream `Eb/N0` (decision D-01). For QPSK, `Es = 2 Eb`; the conversion is
defined once in `modulation/qpsk.py`. Every QPSK symbol has unit magnitude, so one channel-use
column carries total power `Nt`. The AWGN generator takes the per-stream `Es/N0`, not the total
SNR.

The binding decisions D-01 to D-07 and the full assumption register are in the
[mathematical model](docs/mathematical_model.md).

---

## Implemented algorithms

| Component | Status |
|---|---|
| QPSK mapping and demapping | implemented; unit energy, Gray labels, nearest-neighbour decisions |
| MIMO transmit signal | implemented; `x` shape `(Nt, Ns)`, independent streams |
| Flat Rayleigh channel | implemented; `H ~ CN(0,I)`, seeded, batched |
| AWGN with SNR calibration | implemented; `sigma^2 = 1/(Es/N0)`, circularly symmetric |
| Zero forcing | implemented; direct normal-equation solve, no explicit inverse |
| Biased MMSE | implemented; `noise_variance` is `sigma^2`, not an SNR |
| Exhaustive ML | implemented for QPSK at 2x2 and 4x4; optimal reference |
| BER engine | independent replicates, Student-t intervals, per-stream and aggregate counts |
| Condition diagnostics | 2-norm condition-number summaries for ZF reporting |
| SIC / learned detectors | not implemented |

### Detection formulas

Zero forcing:

```
G_ZF = (H^H H)^{-1} H^H,        x_hat = G_ZF y.
```

Implementation solves `(H^H H) x = H^H y` directly. It requires `Nr >= Nt` and full column rank.
The model requires the channel condition number alongside ZF results; the experiment reports a
10,000-matrix diagnostic sample per system.

Biased MMSE with `Es = 1`:

```
G_MMSE = (H^H H + sigma^2 I)^{-1} H^H.
```

The bias is `a_k = [G_MMSE H]_kk = 1 - [E_MMSE]_kk`, which lies in `[0,1)`: strictly positive for
a full-rank channel and zero for a null stream. For QPSK, the real positive bias does not change
hard decisions. MMSE tends to ZF as `sigma^2 -> 0`.

Exhaustive ML:

```
x_hat_ML = argmin_{x in S^Nt} || y - H x ||^2.
```

The QPSK candidate set has `4^Nt` points: 16 at 2x2 and 256 at 4x4. The implementation loops over
candidates and vectorises over the batch, so memory is `O(batch * Nr)`.

---

## Results

### BER experiment

| Setting | Value |
|---|---|
| Systems | 2x2 and 4x4 |
| Detectors | ZF, biased MMSE, exhaustive ML |
| SNR | per-stream `Eb/N0`, 0 to 20 dB in 2 dB steps for ZF/MMSE |
| ML range | 2x2: 0 to 12 dB; 4x4: 0 to 8 dB |
| Replicates | 20 independent replicates per point |
| Target errors | 200 per replicate, with a 2,000,000-bit safety limit |
| Batch size | 20,000 channel uses |
| Confidence interval | 95% Student-t over replicate BERs |
| Base seed | `20261127` |
| Total simulated bits | 182,960,000 |

Each point uses a deterministic point seed shared by all detectors at that SNR. Replicate seeds
are spawned children of the point seed. Per-stream and aggregate summaries are stored, as required
by D-07. The plots show the aggregate BER with 95% t-intervals.

The aggregate interval is deliberately **not** a single-run binomial Wilson interval. Spatial
streams share the same channel and noise, so aggregate stream errors are correlated. The
independent-replicate design avoids that invalid independence assumption.

### BER table

Values are aggregate BER. `-` means the point was not measured within the error/bit budget.

| System | Detector | 0 dB | 4 dB | 8 dB | 12 dB | 20 dB |
|---|---|---|---|---|---|---|
| 2x2 | ZF | 1.466e-1 | 7.734e-2 | 3.539e-2 | 1.496e-2 | 2.533e-3 |
| 2x2 | MMSE | 1.055e-1 | 5.389e-2 | 2.403e-2 | 9.980e-3 | 1.660e-3 |
| 2x2 | ML | 8.649e-2 | 2.894e-2 | 6.635e-3 | 1.228e-3 | - |
| 4x4 | ZF | 1.465e-1 | 7.756e-2 | 3.527e-2 | 1.514e-2 | 2.452e-3 |
| 4x4 | MMSE | 6.623e-2 | 3.084e-2 | 1.298e-2 | 5.353e-3 | 8.538e-4 |
| 4x4 | ML | 2.664e-2 | 2.465e-3 | 1.077e-4 | - | - |
| reference | ZF closed form (8.4) | 1.464e-1 | 7.714e-2 | 3.546e-2 | 1.506e-2 | 2.481e-3 |

### What the results support

- **ZF validation:** 20 of the 22 ZF points have the exact closed form inside their 95% t-interval.
  The two marginal exceptions are 4x4 at 4 dB (closed form `7.7137e-2` versus interval
  `[7.7138e-2, 7.7983e-2]`) and 4x4 at 16 dB (`6.1638e-3` versus `[6.1701e-3, 6.4430e-3]`). Both
  are within about 0.1% of the interval edge; this is consistent with 95% marginal coverage over
  22 points and shows no systematic offset.
- **Square-array ZF:** the 2x2 and 4x4 ZF intervals overlap at all 11 shared SNR points, as
  predicted for square arrays at fixed per-stream `Eb/N0`.
- **MMSE ordering:** MMSE is below ZF at all 22 shared points. It is biased, but the bias does not
  change QPSK hard decisions for full-rank channels.
- **ML ordering:** ML is below MMSE on every measured ML point. ML is not measured above 12 dB
  (2x2) or 8 dB (4x4) because the target error count is not reached within the bit budget.
- **No diversity claim:** no diversity-order slope is fitted or claimed in this release.

### Channel conditioning

Diagnostic sample: 10,000 i.i.d. channel matrices per system, threshold 100.

| System | Median | 95th percentile | Maximum | Fraction above 100 |
|---|---|---|---|---|
| 2x2 | 2.93 | 10.74 | 153.09 | 0.05% |
| 4x4 | 7.37 | 26.62 | 515.88 | 0.31% |

These summaries are stored in `results/ber_mimo.json` and repeated on each ZF record. They are a
representative diagnostic sample, independent of the BER simulation seeds.

---

## Figures

![2x2 BER versus Eb/N0](results/figures/ber_mimo_2x2.png)

*Figure 1. 2x2 MIMO, uncoded QPSK over i.i.d. Rayleigh fading. Markers are replicate means with
95% Student-t intervals; the dashed line is the exact ZF curve of equation (8.4). ML is measured
to 12 dB.*

![4x4 BER versus Eb/N0](results/figures/ber_mimo_4x4.png)

*Figure 2. 4x4 MIMO, same configuration. ML is measured to 8 dB; above that range the target error
count was not reached within the bit budget, so no high-SNR ML claim is made.*

---

## Installation

Requires Python 3.10 or newer. The project is developed in a conda environment; no other virtual
environment is used.

```bash
git clone https://github.com/Amirhossein-Solati/wireless-phy-learning-lab.git
cd wireless-phy-learning-lab
conda create -n phy-lab python=3.11
conda activate phy-lab
pip install -e ".[dev,plot,config]"
```

Run the checks:

```bash
ruff check src tests experiments
pytest -q
```

At v0.1.0 the suite contains 208 tests. CI runs the same lint and test commands on Python 3.10,
3.11, and 3.12.

---

## Reproducing the experiment

```bash
conda activate phy-lab
python experiments/ber_mimo.py
python experiments/plot_ber_mimo.py
```

The simulation reads `experiments/configs/ber_mimo.yaml`, writes `results/ber_mimo.json`,
`results/ber_mimo.csv`, and `results/ber_mimo_config.yaml`, then the plotting script regenerates
the PNG figures. The full experiment takes about three minutes in the reference environment.

The seed policy is:

```
point seed = SeedSequence([base_seed, system_index, snr_key])
replicate seeds = spawned children of the point seed
```

All detectors at a given system/SNR use the same point seed, so their channel/noise streams are
paired. The exact per-point and per-replicate seeds are stored in the JSON. A clean rerun with the
same configuration reproduces the committed counts.

### Output files

| Path | Contents |
|---|---|
| `experiments/configs/ber_mimo.yaml` | input configuration |
| `results/ber_mimo_config.yaml` | byte-identical copy of the configuration used |
| `results/ber_mimo.json` | config, environment, git revision, condition summaries, aggregate and per-stream summaries, replicate seeds and replicate BERs |
| `results/ber_mimo.csv` | flat aggregate table with condition columns for ZF |
| `results/figures/ber_mimo_2x2.png` | 2x2 figure |
| `results/figures/ber_mimo_4x4.png` | 4x4 figure |

See [`docs/experiments.md`](docs/experiments.md) for the statistical design and schema.

---

## Repository layout

```
src/wireless_phy_lab/
  modulation/qpsk.py        QPSK mapping, demapping, Eb/N0 conversion
  mimo.py                   spatial-multiplexed transmit frames
  channel/awgn.py           calibrated complex AWGN
  channel/rayleigh.py       i.i.d. Rayleigh channel matrices
  detection/linear.py       ZF and biased MMSE
  detection/ml.py           exhaustive QPSK ML
  evaluation/ber.py         error counting and Wilson intervals
  evaluation/monte_carlo.py single-run BER simulation
  evaluation/replicates.py  independent-replicate t-intervals
  evaluation/diagnostics.py condition-number summaries
  evaluation/reference.py   exact AWGN and ZF reference curves
experiments/                experiment and plotting scripts
results/                    committed results and figures
docs/                       mathematical and experiment documentation
tests/                      unit, statistical, and experiment tests
```

---

## Limitations

- No SIC and no learned or model-driven detectors.
- ML is measured only where the target error count is reachable: 2x2 up to 12 dB, 4x4 up to 8 dB.
- No diversity-order slopes, outage analysis, or capacity results.
- i.i.d. Rayleigh fading only; no correlation, keyholes, path loss, or time variation.
- Perfect CSI and known noise variance; no channel estimation.
- Uncoded QPSK only; no LLRs or higher-order QAM.
- 2x2 and 4x4 only; nothing here says anything about large-scale MIMO.
- The committed ZF validation has 20/22 closed-form points inside the 95% intervals; the two
  marginal exceptions are disclosed above.
- All results are simulation. No hardware measurement is claimed.

---

## References

Sources used for the implementation and validation:

1. D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*. Cambridge University Press,
   2005. DOI: [10.1017/CBO9780511807213](https://doi.org/10.1017/CBO9780511807213)
2. A. Goldsmith, *Wireless Communications*. Cambridge University Press, 2005.
   DOI: [10.1017/CBO9780511841224](https://doi.org/10.1017/CBO9780511841224)
3. E. Telatar, "Capacity of multi-antenna Gaussian channels," *European Transactions on
   Telecommunications*, vol. 10, no. 6, pp. 585-595, 1999.
   DOI: [10.1002/ett.4460100604](https://doi.org/10.1002/ett.4460100604)
4. R. Lupas and S. Verdu, "Linear multiuser detectors for synchronous code-division multiple-access
   channels," *IEEE Trans. Inf. Theory*, vol. 35, no. 1, pp. 123-136, 1989.
   DOI: [10.1109/18.42183](https://doi.org/10.1109/18.42183)
5. M. Honig, U. Madhow and S. Verdu, "Blind adaptive multiuser detection," *IEEE Trans. Inf.
   Theory*, vol. 41, no. 4, pp. 944-960, 1995.
   DOI: [10.1109/18.391241](https://doi.org/10.1109/18.391241)
6. H. V. Poor and S. Verdu, "Probability of error in MMSE multiuser detection," *IEEE Trans. Inf.
   Theory*, vol. 43, no. 3, pp. 858-871, 1997.
   DOI: [10.1109/18.568697](https://doi.org/10.1109/18.568697)
7. G. J. Foschini, "Layered space-time architecture for wireless communication in a fading
   environment when using multi-element antennas," *Bell Labs Technical Journal*, vol. 1, no. 2,
   pp. 41-59, 1996. DOI: [10.1002/bltj.2015](https://doi.org/10.1002/bltj.2015)
8. G. D. Golden, G. J. Foschini, R. A. Valenzuela and P. W. Wolniansky, "Detection algorithm and
   initial laboratory results using V-BLAST space-time communication architecture," *Electronics
   Letters*, vol. 35, no. 1, pp. 14-16, 1999.
   DOI: [10.1049/el:19990058](https://doi.org/10.1049/el:19990058)
9. L. Zheng and D. N. C. Tse, "Diversity and multiplexing: a fundamental tradeoff in
   multiple-antenna channels," *IEEE Trans. Inf. Theory*, vol. 49, no. 5, pp. 1073-1096, 2003.
   DOI: [10.1109/TIT.2003.810646](https://doi.org/10.1109/TIT.2003.810646)
10. S. Yang and L. Hanzo, "Fifty years of MIMO detection: the road to large-scale MIMOs," *IEEE
    Communications Surveys & Tutorials*, vol. 17, no. 4, pp. 1941-1988, 2015.
    DOI: [10.1109/COMST.2015.2475242](https://doi.org/10.1109/COMST.2015.2475242)
11. J. W. Craig, "A new, simple and exact result for calculating the probability of error for
    two-dimensional signal constellations," *Proc. MILCOM '91*, pp. 571-575, 1991.
    DOI: [10.1109/MILCOM.1991.258319](https://doi.org/10.1109/MILCOM.1991.258319)
12. NIST Digital Library of Mathematical Functions, §7.2, equations 7.2.1-7.2.2.
    [https://dlmf.nist.gov/7.2](https://dlmf.nist.gov/7.2)
13. J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed. McGraw-Hill, 2008.
14. M. K. Simon and M.-S. Alouini, *Digital Communication over Fading Channels*, 2nd ed. Wiley,
    2005.
15. S. Verdu, *Multiuser Detection*. Cambridge University Press, 1998.

Background and direction of travel, not used as validation sources:

16. S. Deka, K. Deka, N. T. Nguyen, S. Sharma, V. Bhatia and N. Rajatheva, "Comprehensive review of
    deep unfolding techniques for next-generation wireless communication systems,"
    arXiv:2502.05952, 2025.
17. G. Marti, J. Hoydis, F. Ait Aoudia, S. Cammerer and A. Keller, "Sionna Bench: towards
    benchmarks in wireless communications," arXiv:2610.27174, 2026.

---

## Version and license

**Version:** 0.1.0.

**License:** MIT. See [`LICENSE`](LICENSE).

## Author

Amirhossein Solati, M.Sc.
K. N. Toosi University of Technology
