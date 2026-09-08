# No-reset post-repair transfer screen

## Frozen test and validity incident

The affine soft-pairwise architecture and thresholds were frozen after development on
the with-resets acquisition. The maximum-depth no-reset circuit (`circuit_24`, 25
decoding rounds) was then opened, using the same first-60,000-row calibration and
last-40,000-row evaluation rule.

The first execution was invalid: only 4 of 104 measurement signatures matched graph
edges. Investigation found that the circuit-noise builder represented readout error as
a physical X before measurement. That changes the future state when no reset follows;
a readout-classification error must instead flip only the classical measurement
record. A unit test now distinguishes these mechanisms. The invalid artifact is
retained as `results/rigetti_soft_matching_noreset_invalid.json`.

The repair used circuit topology and no model threshold or logical-label feedback.
Nevertheless, because the logical result of the invalid run had been observed, the
corrected test is a **post-repair independent-session transfer**, not a pristine
single-look confirmation.

## Corrected result

| Decoder | Logical error |
|---|---:|
| Circuit template, hard bits | 20.4150% |
| Pairwise graph, hard bits | 27.2050% |
| **Pairwise graph, affine I/Q** | **18.2775%** |

Against the stronger hard control, the affine soft decoder reduces logical error by
2.1375 percentage points or 10.47% relative. The paired 95% interval across 40
contiguous 1,000-row blocks is +1.770 to +2.505 points. All 104 measurement
signatures match; 12 residual probabilities hit the declared floor.

The released soft-pairwise value at 25 rounds is 17.623%. Our result is 0.655 points
higher, so the post-repair screen supports a transferable analog-information mechanism
but does not meet
the 0.20-point exact-reproduction margin on this acquisition. The poor hard-pairwise
result also shows that our disclosed Spitz topology/regularization is not the authors'
unreleased graph implementation.

## Scientific conclusion

Across reset and no-reset sessions, the robust contribution is now specific: a cheap
affine I/Q confidence model can improve a structured decoder when it reweights
circuit-aligned measurement edges. Extra spline capacity and causal calibration
tracking have not improved the logical endpoint. The promising research contribution
is therefore the empirically useful operational analog-to-graph interface and its deployable
compilation—not a generic learned decoder or particle world model.

The current Python reference rebuilds a graph per shot and costs 2.97/3.86 ms p50/p99.
This is not real time. A deployment claim requires fixed-topology incremental weights
or an equivalent compiled decoder, measured end to end.

Machine-readable results are in
`results/rigetti_soft_matching_no_resets_confirmation.json`.
