# Stronger surface-code challenge

Protocol frozen before this campaign's evaluation, 8 September 2026.
Earlier experiments informed this design; they are development evidence.

## Questions and comparisons

Does a small causal controller improve logical error after giving the conventional
decoder correlation information and much stronger static calibration? Does its
advantage survive complete refitting and deployment shifts? Can selecting a graph
before running matching remove the cost of decoding twice?

PyMatching supports two-pass correlated matching, including in its public 2.4
implementation. It must be a comparator and an available decision layer, since our
circuits include depolarizing faults. See the [implementation and algorithm
description](https://github.com/oscarhiggott/PyMatching#correlated-matching).
The motivation for testing adaptation against calibrated controls follows
[DGR](https://arxiv.org/abs/2311.16214). Neither comparison constitutes a reproduction
of DGR. [AlphaQubit 2](https://arxiv.org/abs/2512.07737v2) provides the relevant
accuracy/latency motivation; its published hardware numbers are not directly
comparable to these CPU, completed-shot measurements.

## Frozen data and architecture

- Rotated Z-memory circuits at distances 3 and 5, with rounds equal to distance.
- Nominal regimes: (gate depolarization, measurement/reset flip) =
  (0.001, 0.02) and (0.01, 0.001). Nominal switching probability 0.02 per completed
  shot, symmetric stationary prior. The estimator's transition remains 0.02.
- Ten independent complete fits, numbered 0 through 9. Each uses 65,536 calibration
  shots per regime; 256 x 512 action-fitting shots; 256 x 512 selection shots; and
  256 x 512 evaluation shots per condition. No early stopping on observed gains.
- Seeds derive from SHA-256 of campaign name, root seed 2026090801, replicate,
  distance, role, condition, and generator. Record every seed and reject collisions.
- Calibration fits the same local detector/pair affine evidence model as before.
  Separate action data fits residual heads using detector/pair features, matching
  energies, their difference, and either causal or memoryless regime log-odds.
  Both history settings get the same regularization grid {0.001, 0.01, 0.1}.
- Select each family using only the selection stream, with deterministic tie breaks
  favoring fewer decoder calls, then fewer state variables, then canonical name.
  Preserve all selection scores, full model parameters and data hashes.
  Hash shapes, dtypes and mode arrays as well as detector/label contents. Record
  source hashes and reject source mutation during a fit. Exhausted optimization
  budgets retain the best evaluated point and their non-convergence flag; an affine
  optimizer failure or empty action class fails the run rather than silently
  dropping that candidate.

The static family contains the existing 25 circuit-parameter graphs, plus nine
pooled primitive-DEM probability models at mixing fractions 0.1 through 0.9, with
ordinary and correlated matching: 68 fixed candidates. Additionally, for each
backend, optimize the two circuit noise parameters with bounded Nelder-Mead in
log-probability space, starting from the best calibration-grid point; and optimize
the pooled-DEM fraction with bounded scalar optimization. Both searches use the
first 16,384 calibration shots per regime, at most 32 objective evaluations, and
prespecified bounds [0.0003, 0.03] for circuit probabilities and [0.01, 0.99] for
mixture fraction. Add their four outputs to the common selection stage, yielding
72 static candidates. Record optimizers' evaluation traces and termination status.
These are limited-budget fits, not proofs of an optimal static decoder.
Pool matching *mechanism
probabilities*, never matching energies. A pooled independent-mechanism DEM is an
approximation to a shot-level mixture, not its exact distribution. Static selection
has the same downstream selection labels as adaptive selection. Record the old
ordinary-grid comparator separately.

For each matching backend, the temporal and memoryless families each contain nine
posterior thresholds (0.1 through 0.9) and three fitted energy heads. Include the
selected static decoder as a fallback for the overall compiler. Also separately
select a temporal threshold-only policy: it needs one matching invocation per shot,
because its graph choice is made before decoding. Correlated matching internally
performs two blossom passes; one invocation is not one blossom pass.

## Evaluation conditions

All fits and selections use nominal regimes only. Evaluate frozen programs on:

1. Nominal switching: primary comparison.
2. Both rates multiplied by 0.75.
3. Both rates multiplied by 1.25.
4. Less-separated off-grid regimes: (0.002, 0.015), (0.007, 0.003).
5. Slow switching probability 0.005.
6. Fast switching probability 0.1.
7. Independent modes: switching probability 0.5, a negative control for persistence.
8. Regime changes halfway through the circuit; the two endpoint orientations are
   A-then-B and B-then-A, with a persistent orientation across completed shots.
9. Stationary nominal regime A.
10. Stationary nominal regime B.

Mid-circuit changes use the same noiseless circuit and detector definitions; only
noise-instruction probabilities change at a fixed tick boundary. Hardware leakage,
analog measurements, within-experiment feedback and qLDPC are outside this test.

## Inference and gates

Retain error rates and paired error differences for every independent stream.
The primary uncertainty unit is the complete fit/selection/test replicate: use
Student-t intervals on the ten replicate mean paired differences. Also report
stream-conditional intervals to distinguish shot uncertainty from refit uncertainty.
Use 97.5% two-sided replicate intervals for each of the two primary distances
(Bonferroni family coverage at least 95%); secondary intervals are descriptive 95%.

1. Stronger benchmark GO: upper primary interval of compiled-minus-selected-static
   error is below -0.02 percentage points at both distances; all ten replicate point
   estimates improve. Correlated-only selected policies and baselines are also
   reported explicitly, with ordinary matching as a secondary diagnostic. The
   correlated-only compiler (including correlated static fallback) must also beat
   the correlated static comparator with primary interval upper bound below zero.
2. Temporal evidence GO: selected temporal family beats the selected capacity-symmetric
   memoryless family at both distances using the same primary intervals. Also require
   a matched comparison: for energy heads use the same backend and regularization
   with only history removed; for thresholds select the best memoryless threshold
   using the same backend. These do not prove a unique latent sufficient statistic.
3. Transfer GO: for conditions 2–6, all ten distance/condition upper intervals are
   below +0.05 percentage points, and at least six are below zero. Use Bonferroni
   intervals across ten cells. Report all cells, including independent modes and
   mid-circuit changes, without reclassifying observed failures.
4. Stationary GO: all four upper intervals versus the correctly calibrated endpoint
   of the selected static backend are below +0.02 percentage points, with Bonferroni
   intervals over four cells. This tighter margin may reject the old policy.
5. Engineering GO: the one-decode lane must itself pass the -0.02-pp primary static
   win and all-replicate improvement conditions. Its complete graph-before-decode
   implementation (compiled features and scalar recurrent update) must agree with its
   offline reference on every tested choice and logical decision, take one matching
   decode per shot, and
   has measured complete-path p99 no greater than 1.25 times the selected static
   decoder on the same host. Report batched throughput separately from individual
   service latency, including frontend and Python/native boundaries. For the
   independent-mode control require a 97.5% upper interval below +0.02 pp relative to
   the matched memoryless ablation at each distance; no temporal improvement is
   expected there. Batch group selection may use two batch API calls but decodes each
   record only once. The service measurement uses a causal batch-one implementation.

For the midpoint condition, the mode-informed nominal-endpoint reference is only an
orientation-based reference: neither endpoint represents its complete mixed circuit.

All five gates are required for the complete robustness-and-cost claim. Passing
some gates supports only those individual conclusions. Ten refits are a limited
estimate of training variability, not universal robustness. Gate failure triggers
a new documented development hypothesis and fresh evaluation namespaces; it never
changes this protocol's thresholds or deletes its results.
