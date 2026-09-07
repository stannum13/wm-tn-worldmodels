# Event-triggered soft-decoder routing

## Question

Can the fixed hard decoder handle most shots while the expensive affine soft graph is
invoked only for unusually uncertain records? The causal routing score is the sum of
per-measurement posterior hard-decision error probabilities. Thresholds for route
budgets `0, 1, 2, 5, 10, 20, 50, 100%` are fixed exclusively from the 60,000
calibration rows, then applied to the locked 40,000-row holdout.

This changes only which already-computed decoder output is selected. It does not use
logical labels, decoder disagreement, or evaluation quantiles for routing.

## Accuracy--routing Pareto

| Session | Calibration budget | Actual test routed | Hybrid error | Full soft gain recovered |
|---|---:|---:|---:|---:|
| with resets | 10% | 10.42% | 16.450% | 14.4% |
| with resets | 20% | 20.54% | 16.418% | 21.5% |
| with resets | 50% | 50.67% | 16.208% | 68.0% |
| without resets | 10% | 12.95% | 19.998% | 19.5% |
| without resets | 20% | 24.10% | 19.690% | 33.9% |
| without resets | 50% | 54.41% | 18.895% | 71.1% |

Routing at most roughly one quarter of shots recovers only 21--34% of the available
gain. About half the stream must take the slow path to recover roughly 70%. The simple
uncertainty-sum trigger is therefore a **deployment NO-GO**: decision-relevant soft
corrections are too diffuse to be treated as rare anomalies.

The result also constrains future triggers. A more complicated selector must beat
this curve on a new development partition and include its own latency; selecting based
on hard/soft decoder disagreement would require running the expensive decoder first
and would not save computation.

## Measured reference latency

| Session | Fixed hard p50/p99 | Rebuilt soft p50/p99 |
|---|---:|---:|
| with resets | 23.8/65.8 us | 2.23/3.06 ms |
| without resets | 28.1/70.8 us | 2.99/3.85 ms |

Even the Python hard reference misses the 1.7-us-per-round deployment target, and the
soft path is dominated by rebuilding hundreds of graph edges. PyMatching 2.4 exposes
edge addition but no public in-place weight setter. The next engineering contribution
must use a fixed-topology backend with mutable weights or consume likelihoods directly;
event routing does not remove that requirement.

Machine-readable curves are in `results/rigetti_soft_routing_with_resets.json` and
`results/rigetti_soft_routing_no_resets.json`.
