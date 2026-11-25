# Mathematical Model — Narrowband MIMO Detection (MVP)

**Status:** approved by the PI on 2026-11-22; decisions D-01 to D-07 in §13 are binding. Revised
2026-11-25 to correct the MMSE bias interval in (7.5) and to align the implementation and reporting
notes with the baseline. No decision was reopened.

**Scope:** narrowband spatial-multiplexing MIMO, QPSK, i.i.d. Rayleigh fading, AWGN, perfect CSI,
`Nt = Nr = 2` and `4`, ZF / MMSE / ML detection, BER versus SNR.

**How to read this document.** Every equation carries a status tag. An equation is not
implementation-ready until its status is `[V]`, `[X]`, or an approved `[D]`.

| Tag | Meaning |
|---|---|
| `[V]` | Verified in the cited source at the stated granularity (paper / section / abstract). |
| `[T]` | Standard textbook result; the source was not opened at equation level. |
| `[D]` | Derived in this project; derivation shown; **needs review before use**. |
| `[X]` | Verified numerically in this project; the check and its tolerance are recorded in §10. |
| `[P]` | Pending PI decision; the choice changes downstream results. |

---

## 1. Scope and assumption register

### 1.1 Scope

| Item | MVP value | Notes |
|---|---|---|
| Duplexing | none (single link) | no uplink/downlink distinction |
| Bandwidth | single narrow band | one channel matrix `H` per channel use |
| Antennas | `Nt = Nr = 2`, `Nt = Nr = 4` | rectangular `4x2` used only as a validation case |
| Modulation | QPSK, Gray | `M = 4`, `log2 M = 2` |
| Channel | flat, i.i.d. Rayleigh | no correlation, no keyholes |
| Noise | spatially and temporally white AWGN | circularly symmetric |
| CSI | perfect at receiver | see A13 for the caveat |
| Coding | none | uncoded BER |
| Detectors | ZF, MMSE, ML (reference) | + SIC in a later unit |

### 1.2 Assumption register

A row's failure does not merely "degrade performance"; it invalidates a specific equation listed in
the last column.

| # | Assumption | Affects | If violated |
|---|---|---|---|
| A1 | Perfect CSI at Rx | all detectors | residual interference; ZF hits an error floor. MMSE degrades gracefully. |
| A2 | `h_ij ~ CN(0,1)` i.i.d. | (3.1), (8.4) | `h_k^H P_k^⊥ h_k` is no longer Gamma; (8.4) invalid; diversity can drop if rank-deficient. |
| A3 | Frequency-flat (narrowband) | (2.2) | `H` becomes `H(f)`; every equation applies per subcarrier. |
| A4 | `Nr >= Nt` | (7.1) | `H^H H` singular; ZF undefined. |
| A5 | Fading dynamics (see §3.2) | (8.1) | average BER unchanged; **estimator variance** and outage change. |
| A6 | Uncoded | (8.2)–(8.5) | coded systems need LLRs; the biased-MMSE LLR is wrong. |
| A7 | Symbol-synchronous | (2.2) | timing offsets across antennas; `H` becomes frequency dependent. |
| A8 | Ideal matched filter, unit energy | (4.2) | coloured noise; `sigma^2 I` must become `R_n`. |
| A9 | No IQ imbalance | (4.1) | `E[n n^T] != 0`; circularity fails; image interference. |
| A10 | No CFO | (3.1) | per-symbol phase rotation; coherent combining destroyed. |
| A11 | Equiprobable symbols | (7.8) | MAP != ML; `Q`-function formulas average the wrong distribution. |
| A12 | Spatially white noise | (7.2), (7.5) | must whiten first; otherwise "MMSE" is not MMSE. |
| A13 | Known `sigma^2` and `E_s` | (7.4) | MMSE degrades to a mis-regularised ZF. **"Perfect CSI" does not imply this.** |
| A14 | Unit-average-energy symbols | (6.1) | forgetting this is a constant dB offset, not a slope error. |

**A13 is a scope decision, not a detail.** The original project brief says "perfect CSI". MMSE requires
`H`, `sigma^2`, and `E_s`. The spec must therefore read **"perfect CSI + known noise variance"** and
not silently borrow (A13) under (A1). `[P] D-04`

---

## 2. Signal model

### (2.1) Transmitted vector — spatial multiplexing

$$\mathbf{x}=\sum_{j=1}^{N_t} s_j\,\mathbf{e}_j,\qquad s_j\in\mathcal{S},\qquad \mathbb{E}[s_j s_k^{*}]=E_s\,\delta_{jk}$$

- **Variables.** `x ∈ C^{Nt}` transmit vector at one channel use; `s_j` complex symbol on stream `j`;
  `e_j` the `j`-th canonical basis vector; `S` the constellation (§5.1); `E_s` energy per symbol per
  stream.
- **Units.** `s_j` carries `sqrt(J)`; `E_s` in J. After §2.3, `E_s = 1` and the model is dimensionless.
- **Derivation.** Definition of spatial multiplexing: antenna `j` radiates `s_j` in the same
  time–frequency resource. Uncorrelatedness of streams is an assumption (independent bit streams).
  Note `E[||x||^2] = Nt E_s`: total power grows with `Nt` unless the budget is split deliberately.
  This is the origin of the SNR-convention trap (§6).
- **Reference.** Foschini (1996), DOI `10.1002/bltj.2015` `[V, paper]`; Golden et al. (1999),
  DOI `10.1049/el:19990058` `[V, paper]`.
- **Status.** `[T]`
- **Interpretation.** `Nt` independent data pipes in the same bandwidth. The capacity gain is the
  point of MIMO; the detection problem is the price.

### (2.2) Received vector

$$\mathbf{y}=\mathbf{H}\mathbf{x}+\mathbf{n},\qquad \mathbf{y}\in\mathbb{C}^{N_r},\ \mathbf{H}\in\mathbb{C}^{N_r\times N_t}$$

Expanded per antenna:

$$y_i=\sum_{j=1}^{N_t}h_{ij}s_j+n_i,\qquad i=1,\dots,N_r$$

- **Variables.** `H` complex baseband channel matrix; `h_ij` couples Tx `j` to Rx `i`; `n` noise.
- **Units.** `y` in `sqrt(J)`; `h_ij` dimensionless (voltage gain, path loss absorbed).
- **Derivation.** Passband model + flat-fading premise (§2.4) + ideal matched filtering + symbol-rate
  sampling. Perfect symbol synchronisation absorbs the common delay into a phase folded into `h_ij`.
  This step is an **assumption**, not algebra — see A7.
- **Reference.** Tse & Viswanath (2005), DOI `10.1017/CBO9780511807213`, Ch. 7 (section-level)
  `[V, chapter]`.
- **Status.** `[T]`
- **Interpretation.** Each `y_i` is a superposition of all `Nt` symbols. For stream `k`,
  `h_ik s_k` is signal and `sum_{j != k} h_ij s_j` is **spatial interference** — finite, discrete, and
  not Gaussian. Every linear detector is a different answer to "how do I undo this without amplifying
  noise?".

### (2.3) Physical form and normalisation

$$\tilde{\mathbf{y}}=\tilde{\mathbf{H}}\tilde{\mathbf{x}}+\tilde{\mathbf{n}},\qquad
\mathbb{E}[\|\tilde{\mathbf{x}}\|^2]=P_{\text{tot}}T_s,\qquad
\mathbb{E}[\tilde{\mathbf{n}}\tilde{\mathbf{n}}^{H}]=N_0\mathbf{I}_{N_r}$$

Divide by `sqrt(E_s)` with `E_s = P_tot T_s / Nt`:

$$\mathbf{x}=\tilde{\mathbf{x}}/\sqrt{E_s}\ \Rightarrow\ \mathbb{E}[|x_j|^2]=1,\qquad
\sigma^2 \triangleq N_0/E_s = 1/\rho$$

- **Variables.** `P_tot` total transmit power (W); `T_s` symbol period (s); `N_0` one-sided noise PSD
  (W/Hz = J); `rho = E_s/N_0` dimensionless.
- **Units — this is where most papers are loose.** `P_tot T_s` is W·s = J. After a **unit-energy**
  matched filter, `n = ∫ r(t) g*(t) dt` has `E[|n|^2] = N_0`, so the discrete noise variance is
  *numerically* `N_0` but dimensionally a **joule**, not a watt. `rho` is J/J = dimensionless. Every
  SNR in this project is therefore a ratio of energies carrying no unit.
- **Derivation.** Change of variables; linearity preserves (2.2); only second moments change.
- **Reference.** Goldsmith (2005), DOI `10.1017/CBO9780511841224` (section-level) `[V, book]`.
- **Status.** `[T]`
- **Interpretation.** The joule disappears into a ratio. If anyone asks "SNR in what?", the answer is
  "$E_s$ per stream over $N_0$ per receive antenna, both in joules, after matched filtering".

### (2.4) Narrowband validity condition

$$B \ll B_c \approx \frac{1}{5\tau_{\text{rms}}},\qquad T_s \gg \tau_{\max}$$

- **Variables.** `B` bandwidth (Hz); `B_c` coherence bandwidth (Hz); `τ_rms` RMS delay spread (s);
  `τ_max` maximum excess delay (s).
- **Derivation.** Not derivable — it is the *definition* of the flat-fading regime. The `1/5` is an
  engineering rule of thumb and several variants exist (`1/5` vs `1/50`). Pick one and cite it.
- **Status.** `[T]`, rule-of-thumb `[P] D-06`
- **Interpretation.** This assumption buys a single matrix `H` per channel use. Violating it does not
  break the algebra; it multiplies the number of matrices by the number of subcarriers.

---

## 3. Channel model

### (3.1) i.i.d. circularly-symmetric complex Gaussian entries

$$h_{ij}\sim\mathcal{CN}(0,1)\ \text{i.i.d.},\qquad
p(h)=\frac{1}{\pi}e^{-|h|^2},\qquad
\mathbb{E}[|h_{ij}|^2]=1,\quad \mathbb{E}[h_{ij}^2]=0$$

Amplitude and phase:

$$|h|\sim f_R(r)=2r e^{-r^2}\ (r\ge0),\qquad \angle h\sim\mathcal{U}[0,2\pi)$$

- **Variables.** `h = h_R + j h_I` with `h_R, h_I ~ N(0, 1/2)` i.i.d.
- **Units.** Dimensionless. `E[|h|^2] = 1` means "unit average power gain" — the path loss has been
  normalised away, so a BER curve here is a function of `E_s/N_0` only and cannot be mapped to a
  range in metres without re-inserting the link budget.
- **Derivation.** Many independent scattered paths, no dominant component → CLT → jointly Gaussian
  zero mean. Isotropic scattering → equal variances, zero cross-correlation → circular symmetry.
  Normalisation `E[|h|^2]=1` forces each real part to variance `1/2`. Marginalising the phase of
  `p(r,θ) = (r/π) e^{-r^2}` gives `f_R(r) = 2r e^{-r^2}`.
- **Reference.** Tse & Viswanath (2005) §7.3.8, p. 327 (section-level, equation number unverified)
  `[V, chapter]`; Telatar (1999), DOI `10.1002/ett.4460100604` `[V, paper]`.
- **Status.** `[T]`
- **Interpretation.** The best-case channel: maximum rank, maximum diversity. Deep fades are common —
  `P(|h| < 0.1) ≈ 1 %` — which is why diversity order matters. `i.i.d.` across `(i,j)` is a *separate*
  assumption (no correlation, no keyholes, spacing `>= λ/2`).

### (3.2) Fading dynamics

$$\mathbf{H}(t)=\mathbf{H}_\ell\quad\text{for } t\in[\ell T_c, (\ell+1)T_c)$$

with the ergodic limit `T_c = T_s` (independent `H` per channel use).

- **Units.** `T_c` in s; `f_d = v f_c / c` Doppler spread (Hz); `T_c ≈ 1/(4 f_d)` rule of thumb.
- **Status.** `[P] D-02` — **decision required.**
- **Interpretation, and an important correction.** Equation (8.4) averages over the *marginal*
  distribution of `H` at one channel use. It therefore gives the correct **ensemble-average** BER
  under *both* ergodic and block fading. What block fading changes is (i) the **estimator variance** —
  with `T_c` symbols sharing one `H`, the effective number of independent trials drops by a factor
  `T_c` — and (ii) whether outage is a meaningful metric. It does *not* invalidate (8.4).

### (3.3) Channel energy

$$\mathbb{E}\big[\|\mathbf{H}\|_F^2\big]=N_tN_r$$

- **Derivation.** Linearity of expectation over `Nt Nr` unit-mean terms.
- **Status.** `[T]`

---

## 4. Noise model

### (4.1) Complex AWGN

$$\mathbf{n}\sim\mathcal{CN}(\mathbf{0},\sigma^2\mathbf{I}_{N_r}),\qquad
p(\mathbf{n})=\frac{1}{(\pi\sigma^2)^{N_r}}\exp\!\left(-\frac{\|\mathbf{n}\|^2}{\sigma^2}\right)$$

$$\mathbb{E}[\mathbf{n}\mathbf{n}^{H}]=\sigma^2\mathbf{I}_{N_r},\qquad \mathbb{E}[\mathbf{n}\mathbf{n}^{T}]=\mathbf{0}$$

- **Units.** `sigma^2` in J before normalisation, `= 1/rho` dimensionless after.
- **Derivation.** Spatial whiteness is an assumption (no mutual coupling, no common-mode
  interference). `E[nn^T] = 0` is **circularity** (properness), inherited from the complex baseband
  representation of real passband WGN. Temporal whiteness follows from a unit-energy Nyquist matched
  filter.
- **Status.** `[T]`
- **Interpretation.** `sigma^2` is the only noise parameter the receiver needs. Its reciprocal is the
  SNR. Every 3 dB error in this project lives here (A13, §9).

### (4.2) Real decomposition — the 3 dB trap

$$n_i=n_i^{R}+\jmath n_i^{I},\qquad n_i^{R},n_i^{I}\sim\mathcal{N}(0,\sigma^2/2)\ \text{i.i.d.}$$

$$\mathbf{y}_{\text{re}}=\mathbf{H}_{\text{re}}\mathbf{x}_{\text{re}}+\mathbf{n}_{\text{re}},\qquad
\mathbf{H}_{\text{re}}=
\begin{bmatrix}\mathbf{H}_R&-\mathbf{H}_I\\ \mathbf{H}_I&\mathbf{H}_R\end{bmatrix}$$

$$\operatorname{Cov}(\mathbf{n}_{\text{re}})=\frac{\sigma^2}{2}\mathbf{I}_{2N_r},\qquad
\mathbb{E}\|\mathbf{n}_{\text{re}}\|^2=N_r\sigma^2=\mathbb{E}\|\mathbf{n}\|^2$$

- **Derivation.** `E[|n|^2] = E[(n^R)^2] + E[(n^I)^2]`. Circularity gives
  `E[(n^R)^2] = E[(n^I)^2] = sigma^2/2`. Hence `tr Cov(n_re) = 2Nr · sigma^2/2 = Nr sigma^2` — the same
  total noise energy as the complex model. Nothing is lost; only the bookkeeping changes.
- **Reference.** Tse & Viswanath (2005) Ch. 7 for the complex model; the real decomposition is
  standard in lattice-decoding literature (Yang & Hanzo 2015 `[V, paper]`).
- **Status.** `[T]`; covariance computation `[D]`
- **Interpretation.** The `1/2` is not a unit conversion, it is the split of a complex variance over
  two real dimensions. This single factor is the most common silent bug in MIMO simulators (§9).

---

## 5. Modulation

### (5.1) QPSK constellation

$$\mathcal{S}=\left\{\frac{\pm1\pm\jmath}{\sqrt2}\right\},\qquad M=4,\qquad
E_s=\mathbb{E}[|s|^2]=1$$

- **Derivation.** Two independent BPSK rails with amplitude `±a`; `E[|s|^2] = 2a^2 = 1` gives
  `a = 1/sqrt(2)`.
- **Reference.** Proakis & Salehi, *Digital Communications*, 5th ed., Ch. 4 (book-level) `[T]`
- **Status.** `[T]`

### (5.2) Energy per bit

$$E_s=(\log_2 M)E_b=2E_b$$

- **Status.** `[T]`
- **Interpretation.** Because `E_s = 2E_b`, every QPSK BER number is 3.01 dB from the BPSK number at
  the same `E_b/N_0`. This is a **second, independent 3 dB** (§9, item 4).

### (5.3) Gray mapping

$(b_I, b_Q) \in \{0,1\}^2$ mapped so that nearest neighbours differ in exactly one bit. For QPSK,
sign of the I-rail carries `b_I`, sign of the Q-rail carries `b_Q`.

- **Status.** `[T]`
- **Interpretation.** With Gray mapping, one symbol error costs one bit error, which is what makes
  (8.7) exact rather than an approximation.

---

## 6. SNR conventions

### (6.1) Per-stream SNR

$$\rho \triangleq \frac{E_s}{N_0}=\frac{1}{\sigma^2}$$

- **Units.** Dimensionless (J/J).
- **Status.** `[T]`

### (6.2) Conversions

$$\rho_{\text{tot}}=N_t\,\rho,\qquad \frac{E_b}{N_0}=\frac{\rho}{\log_2 M}=\frac{\rho}{2}\ \text{(QPSK)}$$

> **Correction to be aware of:** `E_b/N_0 = rho/2` holds **only** with unit-average symbol energy and
> QPSK, i.e. `E_s = 2 E_b`. It is a special case, not a general identity.

### (6.3) dB conversions

$$\rho_{\text{tot}}\big|_{\text{dB}}=\rho\big|_{\text{dB}}+10\log_{10}N_t,\qquad
\frac{E_b}{N_0}\bigg|_{\text{dB}}=\rho\big|_{\text{dB}}-3.0103\ \text{(QPSK)}$$

| If a paper plots | To compare with per-stream `E_s/N_0` |
|---|---|
| per-stream `E_s/N_0` | none |
| total SNR `rho_tot` | subtract `10 log10 Nt` |
| `E_b/N_0` | add `3.0103` |
| "SNR" undefined | **not comparable** — this is the most common cause of 3–9 dB disagreement |

- **Status.** `[T]`
- **Recommendation.** Plot **`E_b/N_0`**; keep per-stream `E_s/N_0` and `sigma^2` internal.
  Justification: total bits per channel use scale as `Nt log2 M` while total energy scales as
  `P_tot T_s`, so `E_b = P_tot T_s / (Nt log2 M)` is **independent of `Nt`** — only then are 2x2 and
  4x4 comparable at equal total power. `[P] D-01`

**Trap this exposes.** At fixed per-stream `Es/N0`, ZF BER on a square array does not depend on `Nt`
at all: the 2x2 and 4x4 curves lie exactly on top of each other (verified, §10 V4). On an `Eb/N0`
axis they separate by `10 log10 Nt`, which is the physically meaningful comparison.

---

## 7. Detection

### (7.1) Zero forcing

$$\mathbf{G}_{\text{ZF}}=\left(\mathbf{H}^{H}\mathbf{H}\right)^{-1}\mathbf{H}^{H},\qquad
\hat{\mathbf{x}}_{\text{ZF}}=\mathbf{G}_{\text{ZF}}\mathbf{y}$$

- **Derivation.** Impose `G H = I` (null all interference exactly). This is `Nt^2` equations in
  `Nt Nr` unknowns, solvable iff `rank H = Nt`, which needs `Nr >= Nt` (A4). The minimum-norm solution
  is the Moore–Penrose pseudo-inverse. Equivalently, the least-squares solution of
  `min_x ||y - Hx||^2` via the normal equations `H^H H x = H^H y`.
- **Reference.** Tse & Viswanath (2005) Ch. 8 §8.1 (section-level) `[V, chapter]`; Lupas & Verdú
  (1989), DOI `10.1109/18.42183`, the decorrelating detector `[V, paper]`.
- **Status.** `[T]`
- **Numerical note.** Never form `(H^H H)^{-1}`. Solve `H^H H x = H^H y` by Cholesky or QR. Report the
  condition number of `H` alongside any ZF result.

**Algorithm 1 — ZF detection**

```
input : y (Nr), H (Nr x Nt)
1  A <- H^H H                       # Nt x Nt Hermitian PSD
2  b <- H^H y
3  x <- solve(A, b)                 # Cholesky; never invert A
4  s_k <- slice(x, k)               # per stream
5  b_hat <- sign(Re s_k), sign(Im s_k)
```

Complexity `O(Nr Nt^2 + Nt^3)` per channel use.

### (7.2) ZF post-detection noise

$$\hat{\mathbf{x}}_{\text{ZF}}=\mathbf{x}+\mathbf{e}_{\text{ZF}},\qquad
\mathbf{e}_{\text{ZF}}=\mathbf{G}_{\text{ZF}}\mathbf{n},\qquad
\mathbb{E}\!\left[\mathbf{e}_{\text{ZF}}\mathbf{e}_{\text{ZF}}^{H}\right]=\sigma^{2}\left(\mathbf{H}^{H}\mathbf{H}\right)^{-1}$$

- **Derivation.** `G(Hx + n) = x + Gn` because `GH = I`. Then
  `E[ee^H] = sigma^2 G G^H` and `G G^H = (H^H H)^{-1} H^H H (H^H H)^{-1} = (H^H H)^{-1}`.
- **Status.** `[T]`
- **Interpretation.** `(H^H H)^{-1}` is the **noise amplification** matrix. It blows up exactly when
  `H` is ill-conditioned: ZF inverts small singular values and multiplies the noise by `1/sigma_min^2`.
  This is the mechanism behind ZF's failure, and it is what a condition-number-stratified experiment
  measures directly.

### (7.3) ZF per-stream SNR

$$\gamma_k^{\text{ZF}}=\frac{\rho}{\left[(\mathbf{H}^{H}\mathbf{H})^{-1}\right]_{kk}}
=\rho\,\mathbf{h}_k^{H}\mathbf{P}_k^{\perp}\mathbf{h}_k,\qquad
\mathbf{P}_k^{\perp}=\mathbf{I}-\mathbf{H}_{-k}\left(\mathbf{H}_{-k}^{H}\mathbf{H}_{-k}\right)^{-1}\mathbf{H}_{-k}^{H}$$

- **Variables.** `h_k` the `k`-th column of `H`; `H_{-k}` all other columns; `P_k^⊥` orthogonal
  projector onto the null space of the other columns.
- **Derivation (Schur complement — the step texts skip).** Partition
  `H^H H = [[A, b],[b^H, c]]` with `c = ||h_k||^2`, `A = H_{-k}^H H_{-k}`, `b = H_{-k}^H h_k`. Then
  `[(H^H H)^{-1}]_kk = (c - b^H A^{-1} b)^{-1}`, and
  `c - b^H A^{-1} b = h_k^H (I - H_{-k}(H_{-k}^H H_{-k})^{-1} H_{-k}^H) h_k = h_k^H P_k^⊥ h_k`.
- **Status.** `[D]` for the Schur step; the result `[T]`
- **Interpretation.** `h_k^H P_k^⊥ h_k` is the squared distance from `h_k` to the span of the other
  columns — how much of `h_k` survives after projecting out the interference subspace. If `h_k` nearly
  lies in that span, `gamma_k -> 0`: a **spatial fade**. This geometric reading is the key to the
  diversity-order result (8.6).

### (7.4) MMSE

$$\mathbf{G}_{\text{MMSE}}=\arg\min_{\mathbf{G}}\mathbb{E}\!\left[\|\mathbf{G}\mathbf{y}-\mathbf{x}\|^2\right]
=\left(\mathbf{H}^{H}\mathbf{H}+\frac{\sigma^{2}}{E_s}\mathbf{I}_{N_t}\right)^{-1}\!\mathbf{H}^{H}
\;\xrightarrow[E_s=1]{}\;\left(\mathbf{H}^{H}\mathbf{H}+\sigma^{2}\mathbf{I}_{N_t}\right)^{-1}\!\mathbf{H}^{H}$$

- **Units.** `sigma^2/E_s` is a ratio of energies → dimensionless. This is why MMSE requires `E_s`.
- **Derivation.** With `e = Gy - x`, `E[xx^H] = E_s I`, `E[nn^H] = sigma^2 I`, `x ⟂ n`:
  `E[yy^H] = E_s H H^H + sigma^2 I`, `E[xy^H] = E_s H^H`, `E[yx^H] = E_s H`.
  Stationarity of `J(G) = tr E[ee^H]` w.r.t. `G*` gives
  `G = E_s H^H (E_s H H^H + sigma^2 I)^{-1}`. Applying the **push-through identity**
  `(A^H A + αI)^{-1} A^H = A^H (A A^H + αI)^{-1}` with `α = sigma^2/E_s` yields (7.4).
- **Reference.** Lupas & Verdú (1989) `[V, paper]`; Honig, Madhow & Verdú (1995),
  DOI `10.1109/18.391241` `[V, paper]`; Tse & Viswanath Ch. 8 §8.1 `[V, chapter]`.
- **Status.** `[T]`
- **Interpretation.** A *regularised* inverse. Where ZF divides by a small singular value `sigma_i`,
  MMSE divides by `sigma_i^2 + sigma^2/E_s`, which is bounded. High SNR → ZF; low SNR → matched
  filter. It is MSE-optimal among linear detectors, **not** BER-optimal.

**Algorithm 2 — MMSE detection**

```
input : y (Nr), H (Nr x Nt), sigma2
1  A <- H^H H + sigma2 * I          # regularised Gram
2  b <- H^H y
3  x <- solve(A, b)
4  b_hat <- sign(Re s_k), sign(Im s_k)
```

Complexity `O(Nr Nt^2 + Nt^3)`, identical to ZF. `sigma2` must be *known* (A13).

### (7.5) MMSE error covariance and bias

$$\mathbf{E}_{\text{MMSE}}\triangleq\mathbb{E}[(\hat{\mathbf{x}}-\mathbf{x})(\hat{\mathbf{x}}-\mathbf{x})^{H}]
=\sigma^{2}\left(\mathbf{H}^{H}\mathbf{H}+\sigma^{2}\mathbf{I}\right)^{-1}\qquad (E_s=1)$$

$$\left[\mathbf{G}_{\text{MMSE}}\mathbf{H}\right]_{kk}=1-\left[\mathbf{E}_{\text{MMSE}}\right]_{kk}\in[0,1)$$

- **Derivation.** `G H = I - sigma^2(H^H H + sigma^2 I)^{-1}` using
  `H^H H = (H^H H + sigma^2 I) - sigma^2 I`. Substituting into `E[ee^H]` collapses the cross terms to
  `sigma^2 (H^H H + sigma^2 I)^{-1}`. The eigenvalues of `E` are `sigma^2/(lambda_i + sigma^2)`,
  which lie in `(0,1]`; the value `1` occurs exactly for `lambda_i = 0`. Hence
  `[GH]_kk` lies in `[0,1)`. It is strictly positive for a full-rank `H` and exactly zero for a
  null stream.
- **Status.** `[D]`; numerically `[X]` — `[GH]_kk` real to `8e-15`, in `[0.165, 1)` over 3000 random
  4x4 full-rank channels at 10 dB.
- **Interpretation.** MMSE **attenuates** the desired symbol. This is the bias. It matters for soft
  output / LLR generation (A6), and it is why `[GH]_kk` must not be treated as 1. A null stream has
  zero bias and a degenerate hard decision; the i.i.d. Rayleigh experiment is full-rank almost
  surely, so this edge case does not affect its curves.

**Biased vs unbiased, resolved.** Two conventions exist:

| Variant | Filter | `[GH]_kk` | Standard use |
|---|---|---|---|
| biased | `(H^H H + sigma^2 I)^{-1} H^H` | `a_k = 1 - [E]_kk < 1` | SINR analysis, hard decisions |
| unbiased | `D^{-1}G`, `D = diag(GH)` | `1` | soft output / LLR, per-stream symbol estimate |

For **QPSK, hard decisions are provably identical**: the decision is `sign(Re x)`, `sign(Im x)`, and
for a full-rank `H`, `a_k` is real and strictly positive, so dividing by `a_k` changes no sign. A
null stream has `a_k = 0` and a degenerate decision; that edge case does not occur almost surely
under the i.i.d. Rayleigh model. For general `M`-QAM the thresholds are at `±d, ±3d, ...` and are
**not** scale invariant, so the two variants do differ — that is out of MVP scope (QPSK only) and
must be revisited before 16-QAM. `[P] D-05`

### (7.6) MMSE per-stream SINR

$$\gamma_k^{\text{MMSE}}=\frac{1}{\left[\left(\mathbf{I}+\rho\mathbf{H}^{H}\mathbf{H}\right)^{-1}\right]_{kk}}-1$$

- **Derivation.** Write `x_hat_k = a_k x_k + c_k` with `a_k = [GH]_kk` and `c_k` the disturbance. By the
  orthogonality principle `c_k ⟂ x_k`, so `[E]_kk = (a_k-1)^2 + E[|c_k|^2]`. With
  `a_k = 1 - [E]_kk`:
  `gamma_k = |a_k|^2 / E[|c_k|^2] = (1-[E]_kk)^2 / ([E]_kk - [E]_kk^2) = (1-[E]_kk)/[E]_kk = 1/[E]_kk - 1`.
  Finally `sigma^2 (sigma^2 I + H^H H)^{-1} = (I + rho H^H H)^{-1}`.
- **Status.** `[D]` — verified numerically: relative discrepancy `1.9e-14` against the direct
  computation `|a_k|^2 / (sum_{j != k}|[GH]_kj|^2 + sigma^2 ||g_k||^2)` over 2000 random 4x4 channels.
- **Interpretation.** Bias and interference leakage are not independent: the *same* matrix
  `(I + rho H^H H)^{-1}` controls both. You cannot "fix" the bias for free — dividing by `a_k`
  rescales signal, interference and noise equally, so the SINR is unchanged.

### (7.7) MMSE → ZF limit

$$\lim_{\rho\to\infty}\mathbf{G}_{\text{MMSE}}=\mathbf{G}_{\text{ZF}},\qquad
\gamma_k^{\text{MMSE}}=\gamma_k^{\text{ZF}}\left(1+O(\rho^{-1})\right)$$

- **Derivation.** `(H^H H + rho^{-1}I)^{-1} = (I + rho^{-1}(H^H H)^{-1})^{-1}(H^H H)^{-1}` by Neumann
  series, convergent for `rho > ||(H^H H)^{-1}||`; right-multiply by `H^H`.
- **Status.** `[T]`
- **Interpretation.** At high SNR the two linear detectors are indistinguishable, hence the common
  BER slope. At low SNR MMSE is much better (measured 2x2 QPSK at 0 dB: `0.1591` vs `0.2113`).

### (7.8) ML reference detector

$$\hat{\mathbf{x}}_{\text{ML}}=\arg\min_{\mathbf{x}\in\mathcal{S}^{N_t}}\|\mathbf{y}-\mathbf{H}\mathbf{x}\|^2$$

- **Derivation.** For Gaussian `n` and equiprobable `x`, MAP maximises
  `p(y|x) ∝ exp(-||y-Hx||^2/sigma^2)`; the monotone log and constant `sigma^2` drop out.
- **Reference.** Tse & Viswanath Ch. 3 `[T]`; Yang & Hanzo (2015) §II `[V, paper]`.
- **Status.** `[T]`
- **Interpretation.** The gold standard and the **mandatory** reference curve. Complexity
  `M^{Nt}` = 16 candidates at 2x2, 256 at 4x4 — trivially simulable at MVP scale. If a learned
  detector ever beats ML in this simulator, the simulator is broken.

**Algorithm 3 — exhaustive ML**

```
input : y (Nr), H (Nr x Nt), constellation S (size M)
1  build candidate matrix C: (M^Nt x Nt), every vector in S^Nt
2  for each candidate c:  d_c <- || y - H c ||^2
3  k <- argmin_c d_c
4  x_hat <- C[k] ; bits <- demap(x_hat)
```

Complexity `O(M^{Nt} Nr Nt)`. Memory `O(M^{Nt} Nt)`. Exhaustive is acceptable at MVP scale; sphere
decoding is a later optimisation, not an MVP requirement.

---

## 8. BER

### (8.1) Definition

$$P_b(\rho)=\mathbb{E}_{\mathbf{H}}\!\left[P_b(\rho\mid\mathbf{H})\right]=\int P_b(\rho\mid\mathbf{H})\,p(\mathbf{H})\,d\mathbf{H}$$

- **Derivation.** Law of total probability. The average is over `H` only; noise and data are already
  averaged inside `P_b(rho|H)`.
- **Status.** `[T]`
- **Interpretation.** BER is an average over a random channel, not a property of one channel. Two
  systems with the same average BER can have very different outage behaviour (A5).

### (8.2) AWGN QPSK closed form — primary chain validation

$$P_b=Q\!\left(\sqrt{2E_b/N_0}\right)=Q\!\left(\sqrt{E_s/N_0}\right)=Q(\sqrt{\rho})$$

$$Q(x)=\frac{1}{\sqrt{2\pi}}\int_x^\infty e^{-t^2/2}\,dt=\frac12\operatorname{erfc}\!\left(\frac{x}{\sqrt2}\right)$$

- **Derivation.** Two independent BPSK rails, amplitude `±1/sqrt(2)`, noise variance `sigma^2/2`:
  `P_rail = Q((1/sqrt2)/(sigma/sqrt2)) = Q(1/sigma) = Q(sqrt(rho))`. Both rails identical, so the bit
  error rate equals the rail error rate.
- **Reference.** DLMF §7.2, Eqs. 7.2.1–7.2.2, `https://dlmf.nist.gov/7.2` `[V, fetched]`.
- **Status.** `[T]` for the result; `[X]` for the numerical anchors (§10 V1).
- **Interpretation.** This is the test that validates the whole chain: mapper, noise scaling, SNR
  calibration, demapper, error counter. If it fails, nothing downstream is trustworthy.

### (8.3) ZF conditional BER — exact, not an approximation

$$P_b^{\text{ZF}}(\rho\mid\mathbf{H})=\frac1{N_t}\sum_{k=1}^{N_t}Q\!\left(\sqrt{\gamma_k^{\text{ZF}}}\right)$$

- **Derivation.** After ZF, `x_hat_k = x_k + e_k` with `e_k ~ CN(0, sigma_k^2)` and **no interference
  term** — ZF removed it by construction. For QPSK, the decision on `Re{x_hat_k}` has threshold 0,
  signal level `1/sqrt(2)`, noise std `sigma_k/sqrt(2)`, giving
  `Q((1/sqrt2)/(sigma_k/sqrt2)) = Q(1/sigma_k)`.
- **Status.** `[D]` for the exactness argument (the result is standard `[T]`)
- **Interpretation.** Because the only remaining randomness is Gaussian, **ZF BER needs no Gaussian
  approximation**. Any statement that "ZF curves are Gaussian approximations" is wrong; the
  approximation is required only for MMSE.

### (8.4) ZF average BER — exact closed form for QPSK

With `m = Nr - Nt + 1` (the per-stream spatial diversity order) and `mu = sqrt(rho/(2+rho))`:

$$\boxed{\;P_b^{\text{ZF}}(\rho)=\left(\frac{1-\mu}{2}\right)^{m}
\sum_{k=0}^{m-1}\binom{m-1+k}{k}\left(\frac{1+\mu}{2}\right)^{k}\;}$$

Asymptotically:

$$P_b^{\text{ZF}}(\rho)\sim 2^{-m}\binom{2m-1}{m-1}\,\rho^{-m}$$

Check `m = 1`: `P_b = (1-mu)/2 = (1/2)(1 - sqrt(rho/(rho+2)))`, and `P_b ~ 1/(2 rho)`.

- **Derivation.**
  1. `gamma_k = rho X_k` with `X_k = h_k^H P_k^⊥ h_k`.
  2. `P_k^⊥` projects onto a subspace of rank `Nr - (Nt-1) = m` that is independent of `h_k`
     (i.i.d. Gaussian columns). The squared norm of the projection of a `CN(0,I)` vector onto a fixed
     `m`-dimensional complex subspace is `(1/2) chi^2_{2m}`, i.e. `Gamma(m,1)`. Hence
     `X_k ~ Gamma(m,1)` with density `x^{m-1}e^{-x}/Gamma(m)`, mean `m`. **Exact for every `Nr >= Nt`.**
  3. `P_b = E[Q(sqrt(rho X))]`. Using Craig's formula
     `Q(u) = (1/pi) ∫_0^{pi/2} exp(-u^2/(2 sin^2 theta)) dtheta` and the Laplace transform of
     `Gamma(m,1)`:
     `P_b = (1/pi) ∫_0^{pi/2} (2 sin^2 theta / (2 sin^2 theta + rho))^m dtheta`.
  4. For integer `m` this integral has the closed form above (standard MRC-over-Rayleigh result with
     effective per-branch SNR `rho/2`).
- **Reference.** Craig (1991), DOI `10.1109/MILCOM.1991.258319` `[V, paper]` for the integral form;
  the MRC/BPSK-over-Rayleigh family is in Simon & Alouini, *Digital Communication over Fading
  Channels*, Ch. 5 `[T]`. The `m`-general closed form as stated here is `[D]`.
- **Status.** `[D]` and `[X]` — **verified to 1e-15 against independent numerical quadrature** for
  `m = 1,2,3,4` at 6 and 12 dB, and confirmed by link simulation at `m = 3` (4x2 slope `2.986`,
  §10 V5).
- **Interpretation — two consequences that matter.**
  1. This is a **zero-variance reference curve** for ZF at any `(Nt, Nr)` and any SNR. A simulator can
     be validated against it exactly, with no Monte Carlo at all.
  2. At fixed per-stream SNR on a square array (`m = 1`), ZF BER is **completely independent of `Nt`**.
     The 2x2 and 4x4 ZF curves coincide. That is a property of the model, not a bug.

### (8.5) MMSE BER — Gaussian approximation, and its status

$$P_b^{\text{MMSE}}(\rho)\ \approx\ \mathbb{E}_{\mathbf{H}}\!\left[\frac1{N_t}\sum_{k=1}^{N_t}Q\!\left(\sqrt{\gamma_k^{\text{MMSE}}}\right)\right]$$

- **Derivation (assumption, not algebra).** Replace the true disturbance `c_k` — a discrete
  interference sum plus Gaussian noise — by a Gaussian with the same variance, then apply the `Q`
  function.
- **Reference.** Poor & Verdú (1997), DOI `10.1109/18.568697` `[V, abstract only]`
- **Status.** `[T]`, and **`[P]` for citation** — the body of the paper has not been read (O-2).
- **Validity.** Acceptable when residual interference is a sum of many weak independent contributions
  (large `Nt`), or when noise dominates. **Not** acceptable for tail/floor prediction, for 2x2 at
  moderate SNR, or when the bias `a_k` is far from 1.
- **Measured bias of the approximation** (`[X]`, §10 V3): the approximation is systematically
  **1.5–2 % optimistic** at every point tested, never high. That direction is exactly the
  Poor–Verdú phenomenon.
- **Rule.** A Gaussian-approximation curve is a sanity band, never a validation target and never
  labelled "simulation".

### (8.6) Diversity order — corrected

**Definition.**

$$d\triangleq-\lim_{\rho\to\infty}\frac{\log P_b(\rho)}{\log\rho}$$

i.e. the asymptotic slope of `log P_b` versus `log rho`.

| Detector | Diversity, `Nr >= Nt` | Square `Nt = Nr` | Evidence |
|---|---|---|---|
| ZF | `Nr - Nt + 1` | 1 | derived (8.4); measured slope `0.996` (2x2), `2.986` (4x2, `m=3`) |
| MMSE | `Nr - Nt + 1` | 1 | inherited from (7.7); **not independently measured** (O-3) |
| **ML, spatial multiplexing** | **`Nr`** | **`Nt`** | **measured `1.850` (2x2, 12 M bits), `3.994` (4x4)** |
| ML, full-rate STBC | `Nt Nr` | `Nt Nr` | different system — **not this one** |

> **Correction to the research report.** The report claimed ML diversity `Nt Nr` for this system. That
> is the space–time-coding result, where every symbol traverses all `Nt Nr` paths. Under spatial
> multiplexing each symbol is transmitted from one antenna, so the dominant single-symbol error event
> has a pairwise error probability governed by `||h_j||^2 ~ (1/2) chi^2_{2Nr}`, giving diversity `Nr`.
> A union bound over events differing in `|S|` positions gives diversity `Nr|S|`; the minimum is
> `|S| = 1`, hence `Nr`. Measured: 3.994 at 4x4, which cannot be 16.

**Why linear detectors saturate at slope 1 on square arrays.** With `Nt = Nr` exactly **one** degree
of freedom survives after nulling the interferers. That single leftover dimension is a scalar Rayleigh
channel, so the system behaves like a 1x1 Rayleigh link: diversity 1. Each extra receive antenna adds
one leftover dimension and one order of diversity. ML never nulls anything and exploits all `Nr` paths.

### (8.7) BER vs SER, per-stream vs aggregate

With `p = Q(sqrt(2 E_b/N_0))`:

$$P_s=2p-p^2,\qquad P_b=p,\qquad P_b=1-\sqrt{1-P_s}\ \text{(exact)}$$

$$P_{b,\text{agg}}=\frac{1}{N_t}\sum_{k=1}^{N_t}P_{b,k}$$

- **Derivation.** I and Q rails are independently in error with probability `p`. Exactly one bit wrong
  with probability `2p(1-p)`, two bits with `p^2`. So
  `P_b = (1/2)[2p(1-p) + 2p^2] = p` and `P_s = 2p - p^2`. Hence
  `1 - P_s = (1-p)^2` and `P_b = 1 - sqrt(1-P_s)`.
- **Status.** `[D]`
- **Interpretation.** The common `P_b ≈ P_s/log2 M` is a first-order approximation, **exact only as
  `p -> 0`**. Under i.i.d. channels `E[P_b,agg] = E[P_b,1]`, so per-stream and aggregate BER coincide
  *in expectation* but not realization by realization. `[P] D-07`

---

## 9. The 3 dB trap

### 9.1 Four independent places a factor of 2 appears

| # | Quantity | Correct | Buggy | Effect |
|---|---|---|---|---|
| 1 | complex noise variance | `sigma^2` | `sigma^2/2` | 3 dB optimistic |
| 2 | real per-dimension variance | `sigma^2/2` | `sigma^2` | 3 dB pessimistic |
| 3 | QPSK per-rail energy | `E_s/2` | `E_s` | 3 dB optimistic |
| 4 | `E_s` vs `E_b` | `E_s = 2E_b` | `E_s = E_b` | 3 dB axis offset |

### 9.2 Mechanism

Every BER expression is a function of one dimensionless ratio. Doubling the effective noise sends
`rho_eff -> rho_eff/2`, shifting the whole curve right by `10 log10 2 = 3.0103` dB. **The slope is
unchanged**, so the error is invisible in the shape and detectable only against a numeric anchor.

Two bugs can cancel: per-rail `E_s` (item 3) *and* per-real-dimension `sigma^2` (item 2) leave
`rho_eff` correct. This is why "it works and I do not know why" code exists, and why it breaks the
moment the modulation or the domain changes.

### 9.3 Five checks that catch it

1. **Energy equality.** `E||n_re||^2 = Nr sigma^2` and `E||n||^2 = Nr sigma^2` must match exactly.
2. **SNR invariance.** Compute `rho` in both domains; require relative difference `< 1e-12`.
3. **Anchor.** With `H = 1` (AWGN), the chain must produce `P_b = Q(sqrt(2 E_b/N_0))`.
4. **Cross-domain.** Same detector in complex and real domains must agree within Monte Carlo error.
5. **`Q` vs `erfc`.** MATLAB's `qfunc` and `erfc` differ by `sqrt(2)` in the argument — 3.01 dB.
   Assert one known value (`Q(3) = 1.3498980316e-3`) at start-up.

---

## 10. Numerical anchors (validation targets)

All values are computations of the stated analytical expressions, or link-level Monte Carlo where
marked. They are **targets for our simulator**, not published results.

### V1 — `Q` function `[X]`

| `x` | `Q(x)` | reference value |
|---|---|---|
| 1 | 1.586552539314571e-1 | 1.586552539314457e-1 |
| 2 | 2.275013194817922e-2 | 2.275013194817920e-2 |
| 3 | 1.349898031630096e-3 | 1.349898031630090e-3 |
| 4 | 3.167124183311998e-5 | 3.167124183312000e-5 |

Max relative error `4.4e-15`.

### V2 — AWGN QPSK `[X]` (from (8.2), `P_b = Q(sqrt(rho))`)

| `E_s/N_0` dB | `E_b/N_0` dB | `P_b` |
|---|---|---|
| 0 | -3.010 | 1.58655e-1 |
| 3.010 | 0 | 7.86496e-2 |
| 10 | 6.990 | 7.82701e-4 |
| 12 | 8.990 | 3.43026e-5 |

Inverse anchors: `P_b = 1e-2 -> 7.334 dB`; `1e-3 -> 9.800 dB`; `1e-4 -> 11.409 dB` (in `E_s/N_0`).

### V3 — MMSE, direct link Monte Carlo `[X]` (400 k trials, seed 424242) vs Gaussian approximation

| System | `E_s/N_0` | direct MC | Gaussian approx | ratio |
|---|---|---|---|---|
| 2x2 | 0 dB | 0.15909 | 0.1563 | 1.018 |
| 2x2 | 10 dB | 0.02975 | 0.02918 | 1.019 |
| 4x4 | 0 dB | 0.10782 | 0.1062 | 1.015 |
| 4x4 | 10 dB | 0.01638 | 0.01629 | 1.006 |

The approximation is consistently low by 0.6–1.9 %. Compare with ZF at the same points (exact):
0.211325 and 0.0435645 — MMSE is decisively better, as expected at low SNR.

### V4 — ZF closed form and `Nt` independence `[X]`

| `E_s/N_0` | closed form `m=1` | MC 2x2 (300 k) | MC 4x4 (300 k) |
|---|---|---|---|
| 0 dB | 0.211325 | — | — |
| 10 dB | 0.0435645 | 0.043499 | 0.043562 |
| 20 dB | 0.00492623 | — | — |

2x2 and 4x4 agree within Monte Carlo error, confirming the "no `Nt` dependence" property of (8.4).

### V5 — ZF general `m`: closed form vs independent quadrature `[X]`

| `m` | `E_s/N_0` | closed form | quadrature | rel. error |
|---|---|---|---|---|
| 1 | 12 dB | 2.884491e-2 | 2.884491e-2 | 0 |
| 2 | 12 dB | 2.448087e-3 | 2.448087e-3 | 6.7e-16 |
| 3 | 12 dB | 2.297337e-4 | 2.297337e-4 | 5.6e-16 |
| 4 | 12 dB | 2.259215e-5 | 2.259215e-5 | 1.0e-15 |

### V6 — measured diversity slopes `[X]`

| System | range | measured slope | R^2 | theory |
|---|---|---|---|---|
| ZF 2x2 (closed form) | 20–35 dB | 0.996 | — | 1 |
| ZF 4x2 (`m=3`, closed form) | 20–35 dB | 2.986 | — | 3 |
| ML 2x2 QPSK | 8–20 dB, 12 M bits | 1.850 | 0.998 | `Nr = 2` |
| ML 4x4 QPSK | 12–15 dB, 7.2 M bits | **3.994** | 0.989 | `Nr = 4` |

ML 2x2 reads 1.85 rather than 2.00 because that range is sub-asymptotic; it still excludes `Nt Nr = 4`.

### V7 — motivating gap at 4x4 QPSK, 12 dB `[X]`

ZF closed form `2.8845e-2` versus ML `5.04e-5` — a factor of **572**.

---

## 11. Anti-patterns (do not do these)

| Anti-pattern | Why it is wrong | Correct |
|---|---|---|
| `P_b = Q(sqrt(E[gamma]))` | ignores fading entirely | `E[Q(sqrt(gamma))]` |
| calling the ZF curve a "Gaussian approximation" | after ZF the disturbance *is* Gaussian | it is exact, (8.3) |
| validating a simulator against (8.5) | the approximation is 1.5–2 % off | validate against (8.2), (8.3), (8.4) |
| `(H^H H)^{-1}` then multiply | numerical instability | Cholesky/QR `solve` |
| reporting BER without a CI | no way to judge the point | error-count stopping rule + CI |
| plotting `Es/N0` for 2x2 and 4x4 and reading a ranking | ZF curves overlay exactly | plot `Eb/N0` |
| "perfect CSI" while using `sigma^2` | silently adds an assumption | state A13 explicitly |

Quantified example of the first row: ZF 2x2 at 10 dB gives `Q(sqrt(10)) = 7.83e-4` against the true
`4.36e-2` — a **55x** error, equivalent to ~17 dB.

---

## 12. Traceability: equation -> acceptance test

| Test | Equation under test | Assertion | Tolerance |
|---|---|---|---|
| T-01 | (8.2), V2 | AWGN QPSK BER vs closed form | within CI, 6 SNR points |
| T-02 | (4.2), V1 | `E||n_re||^2 = Nr sigma^2`; `Q(3)` | `1e-12` |
| T-03 | (7.1) | `H = I` reduces ZF to AWGN detection | exact to `1e-12` |
| T-04 | (7.4) | `H = I` reduces MMSE to AWGN detection | exact to `1e-12` |
| T-05 | (7.7) | `G_MMSE -> G_ZF` as `rho -> inf` | rel. error `< 1e-6` at 60 dB |
| T-06 | (7.2) | ZF error covariance vs empirical | within MC CI |
| T-07 | (7.5) | `[GH]_kk` real, in `[0,1)` | imag `< 1e-12` |
| T-08 | (7.6) | MMSE SINR identity vs direct | rel. `< 1e-10` |
| T-09 | (8.4), V4 | simulated ZF BER vs closed form, `(2,2)`, `(4,4)`, `(4,2)` | within CI |
| T-10 | (8.6), V6 | measured slopes ZF `=1`, ZF 4x2 `=3`, ML `=Nr` | slope CI excludes alternatives |
| T-11 | (8.5), V3 | MMSE simulation vs Gaussian approx, expect ~1–2 % optimistic | documented, not asserted equal |
| T-12 | (7.8) | ML `<=` MMSE `<=` ZF at every SNR point | strict, with CI |
| T-13 | §9.3 | all five 3 dB checks | exact |
| T-14 | (8.1) | same seed -> bit-identical output | exact |
| T-15 | (8.7) | `P_b = 1 - sqrt(1 - P_s)` on simulated counts | `1e-12` |

---

## 13. Open decisions

**Status: all seven accepted by the PI on 2026-11-22. These are now binding for the MVP.**

| ID | Decision | Decided | Consequence if reversed |
|---|---|---|---|
| D-01 | SNR axis | `E_b/N_0` on plot, per-stream `E_s/N_0` internal | every figure caption, and 2x2 vs 4x4 comparability |
| D-02 | fading dynamics | ergodic (i.i.d. per channel use) | (8.4) still valid for *average* BER; MC variance and outage reporting change |
| D-03 | ML in MVP | include | no optimal reference, no defensible claim |
| D-04 | spec wording for CSI | "perfect CSI + known noise variance" | MMSE silently assumes `sigma^2` |
| D-05 | MMSE variant | biased; unbiased deferred to 16-QAM | hard decisions identical for QPSK either way |
| D-06 | coherence-bandwidth rule of thumb | fix one and cite it | only affects the scope statement |
| D-07 | BER reporting | per-stream **and** aggregate | hides the distinction that matters once correlation is added |

---

## 14. References

| # | Source | Identifier | Status | Used for |
|---|---|---|---|---|
| 1 | Tse & Viswanath, *Fundamentals of Wireless Communication*, CUP 2005 | DOI `10.1017/CBO9780511807213` | `[V, book/chapter]` | (2.2), (3.1), (7.1), (7.4) |
| 2 | Goldsmith, *Wireless Communications*, CUP 2005 | DOI `10.1017/CBO9780511841224` | `[V, book]` | (2.3), (4.1) |
| 3 | Telatar, "Capacity of multi-antenna Gaussian channels", ETT 10(6):585–595, 1999 | DOI `10.1002/ett.4460100604` | `[V, paper]` | (3.1) |
| 4 | Foschini, "Layered space-time architecture...", BLTJ 1(2):41–59, 1996 | DOI `10.1002/bltj.2015` | `[V, paper]` | (2.1) |
| 5 | Golden et al., "Detection algorithm and initial laboratory results using V-BLAST", EL 35(1):14–16, 1999 | DOI `10.1049/el:19990058` | `[V, paper]` | (2.1), SIC |
| 6 | Lupas & Verdú, "Linear multiuser detectors for synchronous CDMA", IT-35(1):123–136, 1989 | DOI `10.1109/18.42183` | `[V, paper]` | (7.1), (7.4) |
| 7 | Honig, Madhow & Verdú, "Blind adaptive multiuser detection", IT-41(4):944–960, 1995 | DOI `10.1109/18.391241` | `[V, paper]` | (7.4), (7.7) |
| 8 | Poor & Verdú, "Probability of error in MMSE multiuser detection", IT-43(3):858–871, 1997 | DOI `10.1109/18.568697` | `[V, abstract]` | (8.5) caveat |
| 9 | Zheng & Tse, "Diversity and multiplexing", IT-49(5):1073–1096, 2003 | DOI `10.1109/TIT.2003.810646` | `[V, paper]` | (8.6) framing |
| 10 | Yang & Hanzo, "Fifty years of MIMO detection", COMST 17(4):1941–1988, 2015 | DOI `10.1109/COMST.2015.2475242` | `[V, paper]` | (7.8), detector taxonomy |
| 11 | Craig, "A new, simple and exact result...", MILCOM '91, pp. 571–575 | DOI `10.1109/MILCOM.1991.258319` | `[V, paper]` | (8.4) integral form |
| 12 | NIST DLMF §7.2, Eqs. 7.2.1–7.2.2 | `dlmf.nist.gov/7.2` | `[V, fetched]` | (8.2) |
| 13 | Proakis & Salehi, *Digital Communications*, 5th ed., 2008 | book; no DOI | `[T]` | (5.1)–(5.3) |
| 14 | Simon & Alouini, *Digital Communication over Fading Channels*, 2nd ed., 2005 | book; no DOI | `[T]` | (8.4) |
| 15 | Verdú, *Multiuser Detection*, CUP 1998 | **ISBN only, DOI unverified** | `[T]` | background |

---

## 15. Revision history

| Date | Revision |
|---|---|
| 2026-11-22 | Model approved; decisions D-01 to D-07 accepted and binding. |
| 2026-11-25 | Corrected the MMSE bias interval in (7.5) and T-07 from `(0,1]` to `[0,1)`; clarified the null-stream case. Updated the status header. No decision was reopened. |
