# Terminal record-level selective decoding

## Controlled question

Can a fixed pairwise decoder use calibration-average weights for most completed
circuit records and invoke per-shot affine-I/Q weights only for unusually ambiguous
records? Both branches use the **same graph topology**. The selector score is the sum
of per-measurement balanced-head hard-bit surprisal surrogates. It is not an identified
physical readout-error probability.

Thresholds for nominal budgets `0, 1, 2, 5, 10, 20, 50, 100%` are fixed using only
the first 60,000 rows and applied to the locked final 40,000. Logical labels,
hard/soft disagreement, and evaluation quantiles are unavailable to the selector.
Because the score uses the entire completed record, this is terminal-time causal
selection—not early per-round event detection.

## Fixed-topology accuracy Pareto

| Session | Nominal budget | Actual routed | Hybrid error | Benefit vs average weights | Regret vs always-soft | Gain recovered* |
|---|---:|---:|---:|---:|---:|---:|
| with resets | 10% | 10.42% | 16.590% | 0.0875 pp | 0.5275 pp | 14.2% |
| with resets | 20% | 20.54% | 16.508% | 0.1700 pp | 0.4450 pp | 27.6% |
| with resets | 50% | 50.67% | 16.225% | 0.4525 pp | 0.1625 pp | 73.6% |
| without resets | 10% | 12.95% | 25.930% | 1.2750 pp | 7.6525 pp | 14.3% |
| without resets | 20% | 24.10% | 24.940% | 2.2650 pp | 6.6625 pp | 25.4% |
| without resets | 50% | 54.41% | 21.953% | 5.2525 pp | 3.6750 pp | 58.8% |

\*Descriptive ratio of two correlated point differences; it has no interval and is
not the primary evidence. The paired contiguous-row-block intervals for absolute
benefit are stored in the JSON artifacts. At the nominal 20% point, the with-reset
interval crosses zero; the no-reset interval is positive but actual load is 24.10%.

The post-hoc engineering gate is at least 80% of the full soft benefit at no more
than 20% actual routing, with a positive paired-block lower bound. Neither session
passes. This is an **exploratory engineering NO-GO for this uncertainty-sum selector
on these two accessed sessions**, not a general rejection of selective decoding.
Calibration quantiles are workload targets rather than hard rate limits; a real
service would additionally need queueing and an online rate limiter.

## Latency boundary and next architecture

The accuracy curve is counterfactual: both branches were computed for every holdout
record. The current reference reconstructs the soft graph for each record, while a
production implementation should update only 96 (reset) or 104 (no-reset) edge
weights on an immutable graph. Reported soft timings cover graph construction and
decode only; they exclude I/Q inference, scoring, routing, queueing, and data movement.

Upstream PyMatching 2.4 has no public mutable-weight decode API. The next bounded
experiment therefore validates a pinned experimental PyMatching fork with temporary
`edge_reweights` against the reconstruction reference. It must produce zero logical
prediction disagreements and achieve under 100 microseconds amortized per record
before any stronger deployment claim. The ultimate comparison remains the paper's
1.7-microsecond syndrome-round cadence.

Machine-readable curves are in `results/rigetti_soft_routing_with_resets.json` and
`results/rigetti_soft_routing_no_resets.json`.
