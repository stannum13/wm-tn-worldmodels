# Exact temporal-frontier decoder: semantic GO, deployment NO-GO

## Question

Can the fixed Rigetti decoding graph be solved as an exact dynamic program over its
natural temporal separator, avoiding per-record graph reconstruction and a general
matching engine?

This is a deliberately narrow architecture test. It asks whether circuit structure
gives a smaller exact hot path for the already validated analog edge weights.

## Construction and contract

For graph edges \(e\), the decoder chooses \(x_e\in\{0,1\}\), minimizes
\(\sum_e w_e x_e\), and enforces observed detector parities
\(Bx=d\pmod 2\). One extra state bit records logical-observable-zero parity. Edges are
processed when their later detector is introduced in `(time, space, node)` order; a
detector is forgotten only after all incident future edges have been seen.

The claim is exact only for the additive, already graphlike PyMatching objective with
zero or one logical observable. It excludes correlated matching, undecomposed
hyperedges, correction-chain recovery, and multiple logical fault IDs. It is an exact
oracle for this benchmark, not a model of device physics.

Packed weights are keyed to canonical `(endpoints, fault IDs)` identities. The code
rejects ambiguous duplicate edges, self-loops, nonbinary syndromes, and nonfinite
weights.

## Topology audit

The natural order gives a constructive width upper bound, not a proof of minimum
pathwidth.

| session | nodes | edges | retained separator | retained states | transient peak states |
|---|---:|---:|---:|---:|---:|
| with resets, circuit 22 | 92 | 276 | 5 | 64 | 128 |
| without resets, circuit 24 | 100 | 400 | 8 | 512 | 1,024 |

Reset edges span at most one time step. The no-reset graph also has 92 edges spanning
two steps. State count is exponential in separator width.

## Locked real-data differential

Both sessions use 60,000 chronological calibration records and 40,000 untouched test
records from public Ankaa-2 stability-8 data. Per-shot weights use the selected
sample-split affine-I/Q mechanism. The reference reconstructs and decodes a fresh
PyMatching graph for every record.

| session | compiled disagreements | batch disagreements | minimum margin | logical error |
|---|---:|---:|---:|---:|
| with resets | 0 / 40,000 | 0 / 40,000 | 0.000778 | 16.0625% |
| without resets | 0 / 40,000 | 0 / 40,000 | 0.001756 | 18.2775% |

Python and compiled recurrences also agree, including margins, on 200 candidate
records per session. Zero bitwise disagreements over 80,000 records gives a
rule-of-three one-sided 95% empirical upper bound of approximately
\(3.75\times10^{-5}\). Structural tests, not that interval alone, support correctness.

## Latency and decision

Measurements use the same GCP CPU campaign host and exclude compilation.

| session | frontier amortized | batch-one p50 / p99 | mutable matching | record cadence |
|---|---:|---:|---:|---:|
| with resets | 50.77 us | 51.40 / 72.32 us | 34.80 us | 39.1 us |
| without resets | 514.23 us | 522.04 / 706.60 us | 69.80 us | 42.5 us |

**Semantic GO:** this is an independent exact oracle for the selected graph-and-weight
objective and removes reconstruction overhead.

**Deployment NO-GO:** even the reset case loses to mutable Sparse Blossom and misses
record cadence. No-reset is 7.4 times slower than mutable matching and 12.1 times its
cadence budget. Separator growth explains the failure mechanistically.

Retain the decoder as a test oracle and measured pathwidth boundary. Give a
topology-only ordering search one bounded screen: continue only if it lowers the
no-reset transient bag width to at most six. Otherwise prioritize a fused indexed
mutable backend combining affine I/Q calibration, odd-parity edge-weight construction,
and matching in one preallocated compiled call.

Machine-readable results are in `results/rigetti_pathwidth_with_resets.json`,
`results/rigetti_pathwidth_no_resets.json`,
`results/rigetti_frontier_with_resets.json`, and
`results/rigetti_frontier_no_resets.json`. GCP CPU timing is deployment-relevant
evidence, not a substitute for FPGA or prospective hardware measurements.
