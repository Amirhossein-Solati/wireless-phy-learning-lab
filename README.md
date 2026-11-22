# Wireless PHY Learning Lab

A research-oriented simulation framework for the wireless physical layer, focused on
**MIMO detection** with classical linear receivers as a rigorously validated baseline, and
model-driven / learned detectors as the research target.

> **Status: research phase.** The mathematical model is specified and awaiting approval. No
> simulation code exists yet and **no experimental results have been produced**. Nothing in this
> repository should be read as an empirical finding at this stage.

---

## Overview

The project studies the narrowband spatial-multiplexing MIMO detection problem:

```
y = H s + n
```

with `Nt` transmit and `Nr` receive antennas, and asks a deliberately narrow question first:

> Given perfect channel state information and a flat Rayleigh channel, how do zero-forcing, MMSE
> and maximum-likelihood detection actually compare in bit error rate, and what is the *measured*
> diversity order of each?

Only once that baseline is reproducible and validated against closed-form results does the project
move to learned detectors. A learned detector that has not been compared against an exhaustive
maximum-likelihood bound in the same simulator is not evidence of anything.

---

## Research objectives

1. Establish a **validated** classical baseline. Every detector must reproduce an independently
   derived reference curve before it is used to judge anything else.
2. Measure, not assume, the **diversity order** of each detector. Linear receivers are known to
   saturate; the project demonstrates this numerically rather than quoting it.
3. Build a simulation harness whose statistics are defensible: confidence intervals, an error-count
   stopping rule, and paired comparisons across detectors.
4. Then, and only then, ask whether a model-driven detector closes part of the gap to maximum
   likelihood — with a pre-registered hypothesis, so a null result is reportable.

**Non-goals at this stage:** coding, OFDM, channel estimation, hardware, standard compliance.

---

## System model

| Element | Choice |
|---|---|
| Duplexing | single link, no uplink/downlink distinction |
| Bandwidth | narrowband, frequency-flat (one channel matrix per channel use) |
| Antennas | `Nt = Nr = 2` and `Nt = Nr = 4`; `4x2` used as a rectangular validation case |
| Modulation | QPSK, Gray-mapped, unit average symbol energy |
| Channel | i.i.d. circularly-symmetric complex Gaussian, unit average power gain |
| Noise | spatially and temporally white AWGN |
| CSI | perfect at the receiver, **plus known noise variance** |
| Coding | none; uncoded bit error rate |

### Conventions

- Complex baseband. Vectors are bold lowercase, matrices bold uppercase, `(.)^H` is conjugate
  transpose.
- After normalisation, symbol energy is `Es = 1` and the noise variance per complex dimension is
  `sigma^2 = 1/rho`, with `rho = Es/N0` dimensionless.
- Reported BER is an **ensemble average over the channel**, not a property of a single realisation.
- The forward error rate is reported against `Eb/N0`, because only that axis makes `2x2` and `4x4`
  comparable at equal total transmit power.

---

## Mathematical formulation summary

The full derivation, with units, assumptions, and verification status for every equation, is
maintained in the project's internal research notes.

**Received signal**

```
y = H s + n,      H in C^{Nr x Nt},   n ~ CN(0, sigma^2 I),   E[s s^H] = Es I
```

**Detection rules**

```
ZF      G = (H^H H)^{-1} H^H
MMSE    G = (H^H H + sigma^2 I)^{-1} H^H          (Es = 1)
ML      s_hat = argmin_{s in S^Nt} || y - H s ||^2
```

**Post-detection noise and per-stream SNR**

```
E[e e^H] = sigma^2 (H^H H)^{-1}                                  (ZF)

gamma_k^ZF    = rho h_k^H P_k^perp h_k                          (exact)
gamma_k^MMSE  = 1 / [ (I + rho H^H H)^{-1} ]_kk  -  1            (exact identity)
```

**Exact ZF bit error rate for QPSK** (no Gaussian approximation is involved), with
`m = Nr - Nt + 1` and `mu = sqrt(rho / (2 + rho))`:

```
P_b^ZF(rho) = ( (1 - mu)/2 )^m  *  sum_{k=0}^{m-1} C(m-1+k, k) ( (1 + mu)/2 )^k
```

This reduces to `(1/2)(1 - sqrt(rho/(rho+2)))` for a square array and provides a zero-variance
reference curve against which the simulator can be validated exactly.

**Diversity order** (`d = -lim log P_b / log rho`)

| Detector | General | Square array |
|---|---|---|
| ZF | `Nr - Nt + 1` | 1 |
| MMSE | `Nr - Nt + 1` | 1 |
| ML (spatial multiplexing) | `Nr` | `Nt` |

---

## Implemented algorithms

**None yet.** The table records the intended order of work, not completed work.

| Component | Status |
|---|---|
| QPSK mapping and demapping | not started |
| Flat Rayleigh channel generator | not started |
| AWGN generator with SNR calibration | not started |
| Zero forcing detector | not started |
| MMSE detector | not started |
| Exhaustive maximum-likelihood detector | not started |
| BER engine with confidence intervals | not started |

---

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Amirhossein-Solati/wireless-phy-learning-lab.git
cd wireless-phy-learning-lab
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot,config]"
```

Run the test suite:

```bash
pytest
```

---

## Usage examples

Not available yet. Usage documentation will be added together with the first implemented component,
and every example shown here will be executed before it is published.

---

## Experimental results

**No experiments have been run.** This section will only ever contain numbers produced by scripts in
`experiments/` and stored under `results/`, with the configuration and random seed recorded for each
figure.

---

## Figures

No figures yet. Final plots will be placed in `results/figures/`.

---

## References

Sources actually used for the derivations in this project:

1. D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*. Cambridge University Press,
   2005. DOI: [10.1017/CBO9780511807213](https://doi.org/10.1017/CBO9780511807213)
2. A. Goldsmith, *Wireless Communications*. Cambridge University Press, 2005.
   DOI: [10.1017/CBO9780511841224](https://doi.org/10.1017/CBO9780511841224)
3. R. Lupas and S. Verdu, "Linear multiuser detectors for synchronous code-division multiple-access
   channels," *IEEE Trans. Inf. Theory*, vol. 35, no. 1, pp. 123-136, 1989.
   DOI: [10.1109/18.42183](https://doi.org/10.1109/18.42183)
4. M. Honig, U. Madhow and S. Verdu, "Blind adaptive multiuser detection," *IEEE Trans. Inf. Theory*,
   vol. 41, no. 4, pp. 944-960, 1995. DOI: [10.1109/18.391241](https://doi.org/10.1109/18.391241)
5. H. V. Poor and S. Verdu, "Probability of error in MMSE multiuser detection," *IEEE Trans. Inf.
   Theory*, vol. 43, no. 3, pp. 858-871, 1997. DOI: [10.1109/18.568697](https://doi.org/10.1109/18.568697)
6. G. J. Foschini, "Layered space-time architecture for wireless communication in a fading
   environment when using multi-element antennas," *Bell Labs Tech. J.*, vol. 1, no. 2, pp. 41-59,
   1996. DOI: [10.1002/bltj.2015](https://doi.org/10.1002/bltj.2015)
7. G. D. Golden, G. J. Foschini, R. A. Valenzuela and P. W. Wolniansky, "Detection algorithm and
   initial laboratory results using V-BLAST space-time communication architecture," *Electronics
   Letters*, vol. 35, no. 1, pp. 14-16, 1999.
   DOI: [10.1049/el:19990058](https://doi.org/10.1049/el:19990058)
8. L. Zheng and D. N. C. Tse, "Diversity and multiplexing: a fundamental tradeoff in multiple-antenna
   channels," *IEEE Trans. Inf. Theory*, vol. 49, no. 5, pp. 1073-1096, 2003.
   DOI: [10.1109/TIT.2003.810646](https://doi.org/10.1109/TIT.2003.810646)
9. S. Yang and L. Hanzo, "Fifty years of MIMO detection: the road to large-scale MIMOs," *IEEE
   Communications Surveys & Tutorials*, vol. 17, no. 4, pp. 1941-1988, 2015.
   DOI: [10.1109/COMST.2015.2475242](https://doi.org/10.1109/COMST.2015.2475242)
10. J. W. Craig, "A new, simple and exact result for calculating the probability of error for
    two-dimensional signal constellations," *Proc. MILCOM '91*, pp. 571-575, 1991.
    DOI: [10.1109/MILCOM.1991.258319](https://doi.org/10.1109/MILCOM.1991.258319)

Recent work that frames the direction of travel:

11. S. Deka, K. Deka, N. T. Nguyen, S. Sharma, V. Bhatia and N. Rajatheva, "Comprehensive review of
    deep unfolding techniques for next-generation wireless communication systems," arXiv:2502.05952,
    2025 (rev. 2026).
12. G. Marti, J. Hoydis, F. Ait Aoudia, S. Cammerer and A. Keller, "Sionna Bench: towards benchmarks
    in wireless communications," arXiv:2610.27174, 2026.

---

## Limitations and future work

**Limitations**

- Perfect CSI is assumed, including knowledge of the noise variance. No channel estimation.
- Narrowband only. No delay spread, no frequency selectivity.
- QPSK only. Higher-order QAM makes the biased/unbiased MMSE question material, and it is deferred.
- Uncoded transmission. No forward error correction, therefore no soft-output requirement.
- Antenna counts of 2 and 4. Nothing here says anything about large-scale MIMO.

**Future work, in intended order**

1. Successive-cancellation detection (V-BLAST style), as an intermediate baseline.
2. Higher-order QAM and larger arrays, where the linear-to-optimal gap becomes large.
3. Sphere decoding, for a complexity-performance Pareto analysis.
4. Spatially correlated channels.
5. Model-driven detectors (deep unfolding) benchmarked against the maximum-likelihood bound.
6. Imperfect CSI and deep-learning channel estimation.
7. MIMO-OFDM with standardised channel models, and integration with Sionna.

---

## License

Not yet chosen. No license is granted at this time.

## Author
Amirhossein Solati
M.sc
KNTU
