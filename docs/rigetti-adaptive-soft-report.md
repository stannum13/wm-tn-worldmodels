# Does predictable I/Q drift improve logical decoding?

## Mediation test

The I/Q diagnostic established that a rolling affine calibrator predicts recorded hard
decisions better than a static one. This experiment asks whether that gain reaches the
logical endpoint. The pairwise graph remains fixed. On the same locked final 40,000
rows, the existing static calibrator is compared with a calibrator refitted every
5,000 rows using only the preceding 20,000. Neither calibration path sees logical
labels.

## Result

Static affine soft matching has 16.0625% logical error; rolling calibration has
16.0525%. The reduction is only 0.010 percentage points, or 0.062% relative. The
paired interval across eight 5,000-row blocks is -0.077 to +0.097 points. This misses
the 1% relative GO condition by more than an order of magnitude.

The unoptimized rolling refit costs 1.34 s p50 and 1.37 s p99 per 5,000-row update.
That work is outside the hot decoder, but its logical value is unresolved and does not
justify optimization.

## Decision

This branch is a **logical-endpoint NO-GO** despite a positive calibration-prediction
result. It demonstrates why proper-score improvements are not sufficient evidence for
control utility. A robust EKF might reproduce the rolling calibrator more cheaply, but
there is no established logical effect for it to preserve. Switching and particle
filters are therefore stopped at this interface unless new data show an abrupt or
multimodal regime with preregistered logical headroom.

The selected real-data architecture remains a static affine I/Q confidence head that
reweights a fixed graph. The next high-value work is compiling that path and confirming
it on the untouched no-reset acquisition, rather than adding estimator capacity.

Machine-readable results are in `results/rigetti_adaptive_soft_matching.json`.
