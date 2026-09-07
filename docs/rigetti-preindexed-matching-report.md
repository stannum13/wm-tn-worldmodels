# Graph-preindexed mutable matching: incremental gain, optimization-gate NO-GO

## Question

After fusing analog inference and edge-weight construction, how much time is spent
reparsing the same 96/104 endpoint pairs and rediscovering their graph slots on every
record?

The experimental backend compiles fixed endpoints once into a graph-owned Tier-1
plan. Each decode then accepts only a contiguous weight vector, validates it against
the existing normalization range, writes cached directed slots, runs the unchanged
Sparse Blossom decoder, and restores exact discretized base-slot values. It rejects
negative or integral-normalized graphs, ambiguous slots, duplicate endpoints, stale
plans, invalid weights, and nested reweight operations rather than falling back to
graph regeneration.

The exact backend source is archived at
[`stannum13/PyMatching@f53805b`](https://github.com/stannum13/PyMatching/commit/f53805b6acbadadc13dc314c9791912c99313747),
based on Allenator/PyMatching commit
`435dc7ec85c10314c09f069a3d924d3a3dee8251`. The base-to-patch diff SHA-256 is
`d5bc0de2dad9c1c3c6c4bbeaa02c7cf9a0c8fc79f161be7432657b6de362c7fe`.

## Correctness evidence

The full previously designated 40,000-record partition from each Ankaa-2 session was
decoded through both the endpoint and preindexed APIs.

- 0/80,000 logical-prediction disagreements;
- zero maximum returned-solution-weight difference;
- zero differences on 100 ordinary baseline syndromes compared before and after each
  complete session pass;
- 84/84 focused reweighting tests pass on GCP;
- 163 matching tests pass locally, with 24 optional skips.

The test suite includes invalid endpoint dtype/range, duplicates, failed recompilation,
graph mutation, search-graph materialization, invalid weights, success restoration,
and restoration after a decoder exception. The experiment does not claim that the
100-syndrome audit observes every internal byte; equivalence instead rests on both
APIs writing the same discretized weight
\(2\operatorname{round}(wC/2)\) to the same directed slots and on targeted tests of
that invariant.

## Performance

Matching-only timings intentionally give the preindexed API a pre-extracted contiguous
weight vector while the control receives endpoint/weight triples. This isolates the
endpoint parsing and slot-discovery cost. Within each separate timing pass, AB/BA
ordering balances which API sees the immediately repeated syndrome first.

| session | endpoint matching p50 / p99 | preindexed matching p50 / p99 | matching reduction | endpoint fused pipeline p50 | preindexed fused pipeline p50 / p99 | completed-record cadence |
|---|---:|---:|---:|---:|---:|---:|
| with resets | 36.10 / 93.29 us | **30.80 / 87.39 us** | 14.69% | 40.92 us | **36.95 / 93.75 us** | 39.1 us |
| without resets | 45.50 / 128.48 us | **40.13 / 123.98 us** | 11.81% | 50.97 us | **46.76 / 128.75 us** | 42.5 us |

Pipeline reductions are 9.70% and 8.27%. The current fused kernel still emits triples
and hands the preindexed call a strided `[:, 2]` view, so this is not yet a native
weights-only fused pipeline.

**Predeclared optimization-screen NO-GO:** both matching reductions miss the required
20%. The speedup is real and the reset median is 2.15 us below its cadence-derived
budget, but that is only a descriptive median compute screen on one shared CPU run.
The no-reset median remains 4.26 us above cadence, and both p99 values fail badly.
This is neither stable-throughput nor deployment evidence.

## Next bounded experiment

The remaining avoidable work is one Python crossing, one vector/correction allocation,
and restoring all slots after every record. A `decode_batch_preindexed(S, W)` C++ path
can overwrite slots between rows and restore base values once per block, reducing
slot writes from roughly \(2BE\) to \(BE+E\). It preserves the exact discretized MWPM
objective.

Continue only with zero differential disagreements and at least 20% pipeline-throughput
improvement or cadence crossing on both sessions. Batch queueing and oldest-record
latency must still be reported; earlier microbatch results already show why amortized
throughput alone is insufficient.

Machine-readable results are `results/rigetti_preindexed_with_resets.json` and
`results/rigetti_preindexed_no_resets.json`.
