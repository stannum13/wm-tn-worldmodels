# Real Ankaa-2 I/Q calibration screen

## Scope

This experiment is the campaign's first real-hardware streaming primitive. It uses
the public Rigetti Ankaa-2 fast-feedback record released with Caune et al. The file is
5,582,749 bytes, its MD5 is `3b2503a80f2b92916660489e2f07e880`, and Zenodo marks
the record CC BY 4.0. It contains 10,000 prepared-zero and 10,000 prepared-one
calibration measurements of qubit 50, including raw complex I/Q and the hardware hard
decision.

This is a measurement-calibration test, not yet a QEC-decoding result. The first 60%
of each prepared-state trace train the heads; the final 40% form a chronological
holdout. Because both traces come from one acquisition session, uncertainty intervals
across contiguous 200-shot blocks are descriptive and must not be read as
cross-session confidence intervals.

## Matched models and result

The hardware-bit baseline learns two reliability probabilities on the training set.
The two raw-I/Q heads use the same real and imaginary inputs and the same logistic
objective. The affine head has three parameters. The additive piecewise-linear
KAN-inspired head has six fixed knots per input and 13 parameters.

| Head | Parameters | Error | Brier | NLL | 10-bin ECE | GCP vectorized time/sample | Python batch-one p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibrated hardware bit | 2 | 4.475% | 0.04265 | 0.18157 | 0.00272 | recorded | – |
| Linear logistic I/Q | 3 | 4.300% | 0.03756 | 0.14933 | 0.03045 | 49.7 ns | 57.2 us |
| Spline/KAN logistic I/Q | 13 | 4.238% | 0.03531 | 0.13589 | 0.00705 | 343.8 ns | 71.6 us |

Relative to the linear I/Q head, the spline head reduces Brier loss by 0.00226
(6.0% relative; descriptive paired block interval 0.00036–0.00415) and NLL by 0.01344
(9.0%; 0.01014–0.01674). Its 0.0625 percentage-point classification-error reduction
has an interval spanning zero (-0.202 to 0.327 percentage points).

## Decision

This is a narrow **GO for a nonlinear slow calibration head feeding soft information
to a decoder or denoiser**. The gain is in probability quality, not established hard
classification accuracy. That distinction matters: better calibrated uncertainty can
improve downstream soft decoding, particle weighting, probe selection, and abstention
even when the 0.5 decision boundary barely moves.

It is a **NO-GO as a Python hot-path claim**. The vectorized numbers measure throughput,
while the batch-one p99 includes Python call overhead and exceeds the paper's
microsecond-scale FPGA regime. A deployment claim requires compiled batch-one kernels,
end-to-end queue measurements, and independent-session testing.

The next real benchmark reconstructs measurement order from the embedded Stim circuit
and tests whether the improved soft probabilities lower logical-observable error. The
5.6 MB file is adequate for schema validation but contains only 8,000 QEC shots from
one session; the 130 MB `stability_9_raw_data.h5` file is the smallest serious
confirmation set.

Machine-readable records are in
[`results/rigetti_iq_benchmark.json`](../results/rigetti_iq_benchmark.json). The data
can be checksum-verified with
[`scripts/fetch_rigetti_fast_feedback.sh`](../scripts/fetch_rigetti_fast_feedback.sh).
