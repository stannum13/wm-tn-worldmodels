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

Machine-readable results are in `results/rigetti_soft_matching.json`. The method is
anchored to [Caune et al.](https://www.nature.com/articles/s41467-026-73331-6),
[Spitz et al.](https://arxiv.org/abs/1712.02360), and
[Pattison et al.](https://arxiv.org/abs/2107.13589).
