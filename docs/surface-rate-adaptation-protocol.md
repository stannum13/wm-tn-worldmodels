# Follow-up: infer the speed of the nuisance process

8 September 2026. Designed after inspecting the stronger challenge; its results
remain unchanged. Use fresh generator namespaces and report this follow-up separately.

The frozen 0.02-switch filter helps persistent regimes but harms independent modes.
Test whether eight bounded probabilities over `(switching-rate hypothesis, regime)`
can retain the nominal gain and stop trusting memory when persistence disappears.
This is a finite-state Bayesian filter, not filtering over physical error strings.

Freeze the ten distance-5 evidence models from the stronger challenge, and use the
same two correlated matching endpoints. No new evidence fitting is performed, so
inference is conditional on those previously fitted models. The ten independently
fitted models and fresh selection/evaluation streams remain the replicate units.

Three controller families get identical graph and detector features: memoryless,
fixed-switch 0.02, and a joint filter over rates `{0.005, 0.02, 0.1, 0.5}`. The rate
hypothesis persists with probability 0.998 and otherwise refreshes uniformly.
All eight joint state probabilities start at 1/8. The new mode is drawn using the
new rate hypothesis, then current evidence is incorporated exactly once. Numerical
emissions are clipped to `[1e-15, 1-1e-15]`. No learned parameter reads test labels.

For each evidence model, use 256 x 512 fresh nominal selection shots to choose a
threshold from `{0.1,...,0.9}` separately for each family. Use root seed 2026090804.
Evaluate 256 x 512 fresh shots for each of nominal, slow, fast, independent modes,
mid-circuit changes and both stationary regimes. All controller rules remain frozen;
the posterior rate estimate is the only online adaptation. The midpoint condition
is retained as a known out-of-family stress test; this follow-up does not add
midpoint graph templates.

Primary conditions, all required:

1. Nominal: rate-adaptive minus fixed-switch upper 97.5% replicate interval <= +0.02
   percentage points, and rate-adaptive versus the previously selected static
   baseline upper bound below zero.
2. Independent modes: rate-adaptive versus fixed-switch upper 97.5% interval below
   zero, and versus matched memoryless upper bound <= +0.02 percentage points.

The 97.5% per-condition intervals provide Bonferroni coverage across the two primary
noise conditions. Multiple required comparisons within each condition form an
intersection-union test. Report 95% descriptive intervals on slow/fast/stationary/
midpoint controls. Require exact reduction to the existing filter with one rate,
agreement with exhaustive short hidden-state paths, causality and boundedness tests.
State storage is eight float64 probabilities (64 bytes per stream) versus one
posterior scalar (8 bytes); constants and inference-time arithmetic are additional.
Latency of the new eight-state program requires a separate measurement and cannot
inherit the one-state native frontend's result.
