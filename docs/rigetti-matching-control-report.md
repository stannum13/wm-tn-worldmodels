# Ankaa-2 PyMatching structural control

## Why this control is necessary

The learned soft, static graph-feature, and causal Markov decoders all missed the
released stability-9 MWPM result. The embedded Stim circuit contains no calibrated
noise instructions, so its detector error model is empty. This control inserts an
explicit uniform circuit-level noise hypothesis: bit flips before measurement and
after reset, one-qubit depolarization after one-qubit gates, and two-qubit
depolarization after CZ gates. Stim converts that hypothesis to a detector error model
and PyMatching enforces global error-chain consistency.

Five uniform probabilities from 0.002 to 0.05 were compared only on the first 60,000
chronological shots. Probability 0.002 had the lowest training logical error and was
then evaluated on the locked last 40,000 shots.

## Result

The test logical error is **38.7475%**, with a descriptive interval of 37.809–39.686%
across 40 contiguous 1,000-shot blocks. The released
`lep_mwpm_stability_9.txt` value at 27 rounds is **38.819% ± 0.154%**. Our value is
0.0715 percentage points lower, much too small—and methodologically too
non-independent—to claim an improvement. It is a successful reproduction-scale
control.

The 467-term approximate detector error model takes 70.7 us per shot in vectorized
GCP CPU replay. Python batch-one p50/p99 are 85.7/327.8 us. These timings cover a full
27-round record, not streaming FPGA operation, and cannot be compared directly with
the paper's sub-microsecond mean per-round hardware decoder.

## Architectural consequence

Global matching structure recovers the gap that generic learned history models did
not. The base architecture is therefore reversed: PyMatching (or an equivalent
compiled graph decoder) owns the decision hot path. Learned KAN, particle, or drift
models may update the detector error model and soft edge weights asynchronously, but
must prove incremental benefit against this matching control under identical syndrome
access.

The next effect-size target is at least 1% relative logical-error reduction versus
38.7475% (absolute reduction at least 0.3875 percentage points), with a positive
paired block interval and no material tail-latency regression. A type-specific
measurement/one-qubit/two-qubit noise calibration is the cheapest next mechanism
test. A claim beyond this session requires an independent acquisition.

Machine-readable results are in
[`results/rigetti_matching_control.json`](../results/rigetti_matching_control.json).
The decoding engine is [PyMatching](https://github.com/oscarhiggott/PyMatching), and
the public hardware study is
[Caune et al.](https://www.nature.com/articles/s41467-026-73331-6).
