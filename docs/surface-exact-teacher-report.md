# Exact teacher: most remaining loss is outside the small memory state

8 September 2026. This is a fresh-stream diagnostic after the stronger campaign,
using its predeclared replicate-0 distance-3 controller. It does not select a
favorable fitted controller after seeing teacher results.

| Method | Fresh nominal LER |
| --- | ---: |
| Selected static correlated matching | 1.99356% |
| Selected compiled adapter | 1.91154% |
| Exact memoryless DEM teacher | 1.76506% |
| Exact causal DEM teacher | 1.71623% |
| Exact teacher with hidden mode revealed | 1.68190% |

The exact causal teacher improves on the selected compiled adapter by 0.19531 pp
(paired stream 95% interval [-0.22590, -0.16473]). However, the exact memoryless
teacher already improves on that adapter by 0.14648 pp. The clean temporal benefit
is only 0.04883 pp ([-0.06620, -0.03146]). Thus the larger remaining opportunity
cannot be attributed solely to better recurrent memory; it includes more accurate
syndrome-to-logical inference and marginalization over fault configurations.

The full-table stationary Bayes risks, summed without sampling uncertainty, are
1.823168% for memoryless inference and 1.747723% when the physical mode is known.
The empirical table above uses one finite correlated stream sample and differs
from these exact stationary expectations. The exact causal stationary risk was
not analytically integrated.

Of 5,226 static failures, 1,655 are correctable by choosing between the two nominal
correlated endpoint predictions. The exact causal teacher corrects 1,280 static
failures, including 286 where neither endpoint prediction is correct. It also
introduces 553 errors on records the static decoder gets right. These are paired
opportunity and harm counts, not disjoint physical-cause labels. They show a limit
of this two-prediction candidate set.

Under independent modes, the exact filter with the correct 0.5 transition agrees
exactly with the memoryless teacher. Retaining the wrong 0.02 transition worsens
logical error by 0.05760 pp ([+0.03856, +0.07665]). Transition mismatch therefore
causes harm even with exact current-syndrome likelihoods; it is not merely a poor
affine classifier artifact.

Two 256-MiB float64 tables provide the joint detector/logical distributions at d3.
Exhaustive small-fault tests, full-mask equivalence across DEM decomposition,
probability-mass checks, Fourier round trips, and independent circuit/DEM parity
moment checks pass. This is a privileged offline diagnostic with exponential
storage, not a learned decoder or a deployment result. Details and the elementary
one-threshold-per-syndrome derivation are in
[the theory note](surface-exact-teacher-theory.md).

The next compression target is a residual action law selected on actual logical
error, informed by the teacher's conditional risks. Increasing recurrent width
without addressing this decoding gap has weak support from these results.

Artifact: `results/surface_exact_teacher.json`.
