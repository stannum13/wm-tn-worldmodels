# Independent science audit: cross-fitted surface router

Audit performed 8 September 2026 after the first execution and before accepting its
result.

## Initial verdict: FAIL

The auditor found that calibration used Stim sampler seeds `base+10/+11`, while the
nominally independent final stream also reached `base+10/+11` through its helper.
Different requested shot counts mean literal row duplication was not assumed, but the
streams were not independently seeded. The initial artifact was discarded and its
selection-integrity claim withdrawn.

The audit also required narrower language: the comparator is a predeclared
25-candidate static PyMatching search, not general SOTA; one split does not include
fit/selection variability; a coarse static grid omits mixture-DEM and continuous
weight optimization; and hardware/deployment claims remain unsupported.

## Correction

Commit `a3e9d70` assigns disjoint sampler namespaces to calibration (+100), residual
fit (+1000), policy selection (+2000), final test (+3000), and stationary controls
(+4000). It also adds detailed memoryless policies to the selection family and final
comparisons, addressing whether richer current-shot features alone explain the win.

The corrected run retains statistically resolved wins over the static, prior causal,
and selected memoryless comparators at both distances, while preserving all claim
limitations in the main report. This is a conditional PASS for the declared synthetic
benchmark instance, not an algorithmic-robustness or SOTA verdict.
