# Ankaa-2 stability-9 soft-decoding replay

## Locked downstream test

The real-I/Q calibration screen found a 6.0% Brier improvement for a spline/KAN head,
but no established hard-classification gain. This experiment tests whether that
probability improvement survives the actual downstream operation: converting 144
measurements into 108 detector parities and predicting the logical observable.

We downloaded the independently packaged 130,236,173-byte
`stability_9_raw_data.h5` file (MD5 `738e37d66de114e7eacac3342d41294d`). It contains
100,000 Ankaa-2 shots at 28 rounds. Per-qubit arrays are placed into the exact
measurement order of the embedded Stim circuit; Stim then reconstructs detector and
logical-observable bits. All reconstructed hard parities match Stim after its fixed
reference offsets.

The first 60,000 chronological shots train nine per-qubit I/Q-to-hardware-bit
calibrators and the logical decoder. The last 40,000 shots are locked test data. Soft
detector probabilities propagate calibrated measurement probabilities through parity
under an explicit conditional-independence approximation. Every logical model is
trained with the same logistic objective.

## GCP CPU result

| Pipeline | Parameters | Logical error | Brier | NLL | Decoder vectorized ns/shot |
|---|---:|---:|---:|---:|---:|
| Hard syndrome + linear decoder | 109 | 42.395% | 0.24053 | 0.67376 | 1,117 |
| Linear-I/Q soft syndrome + linear decoder | 136 | 42.323% | 0.24080 | 0.67434 | 1,173 |
| Spline/KAN-I/Q soft syndrome + linear decoder | 226 | 42.448% | 0.24086 | 0.67447 | 1,227 |
| Spline/KAN-I/Q soft syndrome + spline decoder | 442 | 42.560% | 0.24059 | 0.67386 | 5,544 |

The spline calibration pipeline worsens logical error by 0.125 percentage points
relative to linear calibration. Its first-minus-second paired interval across 39
contiguous test blocks is -0.285 to -0.008 percentage points, so the direction is
inconsistent with benefit. Brier and NLL differences also fail to clear zero. The
extra spline decoder does not rescue the result.

For context, the released `lep_mwpm_stability_9.txt` reports 38.819% ± 0.154% at 27
rounds on this acquisition. Our simple hard-syndrome learner is about 3.6 percentage
points worse, so it is a useful implementation control but not a competitive decoder.

## Decision and architectural correction

This branch is a **NO-GO**. A better local measurement calibration does not imply a
better logical decoder when independent probabilities are repeatedly multiplied
through long parities. Marginal parity probabilities discard which measurements share
a physical error mechanism, and at high detector rates they collapse toward 0.5.

The next decoder must preserve graph structure and joint evidence. The cheapest
direction is a hard-syndrome graph baseline plus a small set of circuit-local pairwise
features, followed by calibrated soft edge weights. Belief matching is the stronger
reference because it runs belief propagation on the detector-error-model Tanner graph
before MWPM; its public implementation explicitly warns that it is over 100 times
slower than optimized PyMatching, reinforcing the multi-rate design rather than an
always-on neural hot path.

No cross-session or live-feedback claim is made. The block intervals are descriptive,
the calibration targets the recorded hard decisions rather than prepared-state truth,
and the replay excludes communication, buffering, and conditional-operation latency.

Machine-readable records are in
[`results/rigetti_stability9_qec_replay.json`](../results/rigetti_stability9_qec_replay.json).
The graph-aware reference is described by
[Higgott et al.](https://arxiv.org/abs/2203.04948) and implemented in
[BeliefMatching](https://github.com/oscarhiggott/BeliefMatching).
