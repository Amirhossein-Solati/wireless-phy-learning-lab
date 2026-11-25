# Changelog

## 0.1.0 — 2026-11-25

First public research baseline.

### Added

- Gray-mapped QPSK with unit average energy and `Es = 2 Eb` conversion.
- Spatial-multiplexed MIMO transmit frames.
- i.i.d. Rayleigh channel and calibrated complex AWGN generators.
- ZF, biased MMSE, and exhaustive QPSK ML detectors.
- Monte Carlo BER engine with independent replicates and Student-t intervals.
- Per-stream and aggregate BER reporting (D-07).
- Channel condition-number diagnostics for ZF.
- Exact AWGN QPSK and ZF Rayleigh reference curves.
- 2x2 and 4x4 BER experiments with committed configurations, seeds, raw counts, and figures.
- Public mathematical model and experiment documentation.

### Validation

- 208 tests pass.
- ZF closed form agrees with simulation at 20 of 22 ZF points within 95% intervals; the two
  marginal exceptions are disclosed in the README and experiment documentation.
- The 2x2 and 4x4 ZF intervals overlap at all shared SNR points.
- MMSE is below ZF at all shared points; ML is below MMSE on every measured ML point.

### Known limitations

- No SIC or learned detectors.
- ML is measured only up to 12 dB for 2x2 and 8 dB for 4x4.
- No diversity-order fitting.
- No correlated or time-varying channels, coding, LLRs, or channel estimation.

### License

MIT.
