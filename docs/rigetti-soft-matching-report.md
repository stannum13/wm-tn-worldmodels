# Cross-fitted analog pairwise matching

## Controlled question

Does per-shot I/Q confidence improve logical decoding after the hard pairwise graph has
been fixed? The first 60,000 HDF5 rows fit both the label-free Spitz graph and one
balanced affine logistic I/Q classifier per measured qubit. The classifiers predict
the existing hardware hard decision; they never see the logical observable. The last
40,000 rows are locked evaluation data.

For each shot, the average measurement-error contribution is factored out of its
matched graph edge and replaced with the shot's posterior hard-decision error. All 96
measurement locations map to a circuit-derived graph edge; none are discarded. Three
residual edge probabilities reach the declared numerical floor.

## Result

| Decoder | Logical error |
|---|---:|
| Circuit template, hard bits | 16.5150% |
| Pairwise graph, hard bits | 16.6775% |
| **Pairwise graph, affine I/Q** | **16.0625%** |

Against the stronger hard template, analog reweighting reduces absolute error by
0.4525 percentage points, or 2.74% relative. The paired 95% interval over 40
contiguous 1,000-row blocks is +0.048 to +0.857 points. Against hard pairwise, the
reduction is 0.615 points or 3.69%, with interval +0.283 to +0.947 points. The
predeclared 1% mechanism gate passes against both controls.

The identical frozen affine pipeline improves every tested circuit depth:

| Decoding rounds | Hard template | Affine-I/Q pairwise | Relative reduction | Paired 95% interval |
|---:|---:|---:|---:|---:|
| 3 | 32.050% | 27.488% | 14.24% | +4.243 to +4.882 pp |
| 7 | 25.565% | 23.410% | 8.43% | +1.764 to +2.546 pp |
| 11 | 21.683% | 20.423% | 5.81% | +0.872 to +1.648 pp |
| 15 | 20.275% | 19.403% | 4.30% | +0.553 to +1.192 pp |
| 19 | 18.803% | 17.498% | 6.94% | +1.035 to +1.575 pp |
| 23 | 16.515% | 16.063% | 2.74% | +0.048 to +0.857 pp |

Thus the primary gain is not purchased by harming shorter circuits. The released
curve is still lower by more than the 0.20-point reproduction margin at several
intermediate depths, consistent with unavailable prepared-state calibration and graph
implementation details; only the locked deepest validity target passes.

The released soft-plus-pairwise result is 15.901% at 23 decoding rounds. Our locked
holdout is 0.1615 percentage points higher, inside the 0.20-point reproduction gate.
Because the released value may use the same shots for calibration and evaluation and
implementation details are absent, this agreement validates scale and mechanism; it
does not establish superiority or exact reproduction.

## Latency and limitations

The current transparent Python implementation rebuilds a graph for every shot. Its
combined build-and-decode p50/p99 is 2.20/2.80 ms on the GCP CPU, far outside the
paper's real-time budget. The scientific architecture is therefore a reference, not a
deployable hot path. Production work must precompile or incrementally update the
small measurement-edge subset while leaving PyMatching's topology fixed.

The authors' prepared-|0>/|1> calibration data are not released with this file. Our
balanced hardware decisions are pseudo-labels, so the experiment estimates
confidence in the recorded classifier rather than measurement fidelity to the latent
qubit state. Per-shot timestamps are also absent.

## Next gates

1. Compare the affine I/Q head against the already-defined 13-parameter spline/KAN
   head under the same graph and split.
2. Test whether blockwise calibration residuals are predictably time-varying. Start
   with a static state and robust EKF; use a switching model or particles only after a
   multimodality diagnostic.
3. Freeze the winning pipeline, optimize graph publication latency, and confirm once
   on the untouched no-reset acquisition.

## KAN-head ablation

The same locked run was repeated with the 13-parameter-per-qubit spline/KAN-inspired
I/Q head. Logical error is 16.2075%, compared with 16.0625% for the affine head. The
affine head's paired advantage is 0.145 percentage points, with interval -0.090 to
+0.380 points across the same 40 row blocks. The interval is unresolved, while the
point estimate favors the cheaper affine model. The spline/KAN branch is therefore a
**NO-GO** at this interface.

Machine-readable results are in `results/rigetti_soft_matching.json`. The method is
anchored to [Caune et al.](https://www.nature.com/articles/s41467-026-73331-6),
[Spitz et al.](https://arxiv.org/abs/1712.02360), and
[Pattison et al.](https://arxiv.org/abs/2107.13589).
The spline ablation is in `results/rigetti_soft_matching_spline.json`; per-circuit
artifacts use `results/rigetti_soft_matching_circuit_*.json`. All retain block-level
errors for direct paired comparisons.
