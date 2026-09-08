# Causal rolling pairwise calibration

## Test

The hard pairwise graph helped some circuit depths but not the locked 23-round
endpoint. To test the simplest drift explanation, this experiment compares:

- the fixed circuit-derived template;
- one static pairwise graph fitted to the first 20,000 HDF5 rows; and
- a causal rolling graph refitted every 1,000 rows using only the preceding 20,000.

All 80,000 subsequent rows are evaluated. Graph fitting uses detector bits only and
never sees logical labels. Each update is published before decoding the following
block. The file has no per-shot timestamps, so this is a row-order responsivity test,
not proof of wall-clock drift tracking.

## Result

| Model | Logical error |
|---|---:|
| Fixed circuit template | **16.8125%** |
| Static pairwise | 16.8638% |
| Rolling pairwise | 16.9188% |

Rolling minus fixed-template performance is -0.106 percentage points of improvement
(95% paired interval -0.319 to +0.107 pp). Rolling versus static pairwise is -0.055 pp
(-0.168 to +0.058 pp). Both comparisons miss the 1% relative GO threshold and neither
interval establishes benefit.

The rolling graph update takes 69.9 ms p50 and 73.1 ms p99 per 1,000-shot block on the
GCP CPU, or 69.7 us amortized per shot. The unchanged PyMatching batch decode takes
30.6/32.9 us per shot at block p50/p99. These are offline Python measurements, not a
hardware deadline claim.

## Decision

Naive rolling refits are a **NO-GO**. This is evidence against “more responsive” as a
sufficient mechanism, not against time-varying noise generally. Particle filtering is
not justified at this point: there is neither a validated latent transition model nor
evidence of a multimodal graph-state posterior.

The next test isolates the missing access advantage in the published strongest
decoder—per-shot analog I/Q confidence—on a fixed graph. Only residual, predictable
block-to-block graph variation after that control can justify an EKF, switching model,
or slow-lane particle filter.

Machine-readable results are in `results/rigetti_adaptive_pairwise.json`.
