# Causal I/Q drift diagnostic

## Question

After analog confidence improves logical decoding, is the calibration state
predictably time-varying? On the 23-round circuit, a static affine head trained on the
first 20,000 HDF5 rows is compared with an affine head refitted every 5,000 rows using
only the preceding 20,000. Each head predicts recorded hardware hard decisions from
I/Q; logical labels are excluded. Proper scores are evaluated on the next block.

## Result

Across 16 future blocks, rolling calibration reduces Brier loss from 0.0122320 to
0.0119279, a 2.49% relative reduction. The mean absolute reduction interval is
+0.000176 to +0.000432. NLL falls from 0.0562721 to 0.0549718, a 2.31% relative
reduction, with interval +0.000698 to +0.001902. Both pass the declared 1% predictive
gate.

Calibration functions evaluated on a fixed anchor set have median lag-one correlation
0.623 across dimensions. This supports a persistent row-order state rather than pure
independent fit noise, but it is not a wall-clock claim because per-shot timestamps are
absent.

## Routing decision

This is a mechanism-only GO. The subsequent mediation test found only a 0.062%
relative logical-error reduction with an interval crossing zero. Therefore a robust
EKF has no established logical effect to preserve, and switching or particle models
are stopped at this interface; see the
[adaptive soft report](rigetti-adaptive-soft-report.md).

Machine-readable block scores are in `results/rigetti_iq_drift.json`.
