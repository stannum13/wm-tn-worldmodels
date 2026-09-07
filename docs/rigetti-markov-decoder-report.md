# Causal syndrome Markov decoder screen

This experiment packs the four detector events at each of 27 rounds into one of 16
categorical symbols. A class-conditional generative decoder accumulates the log
likelihood of logical classes zero and one with one table lookup per round. Order zero
tests within-round joint structure; orders one and two add temporal memory. All tables
use fixed unit Dirichlet smoothing and the same 60,000/40,000 HDF5-row split.

| Model | Free parameters | Logical error | Brier | GCP vectorized ns/shot |
|---|---:|---:|---:|---:|
| Categorical order 0 | 31 | 41.890% | 0.24253 | 569 |
| Markov order 1 | 511 | 42.073% | 0.24230 | 601 |
| Markov order 2 | 8,191 | 45.193% | 0.25532 | 662 |

The order-zero model improves on the earlier linear detector head's 42.395% error by
0.505 percentage points, recovering 14.1% of its gap to the released 38.819% MWPM
anchor. It misses the preregistered 20% gate. Temporal order does not help: order one
worsens threshold error, and order two overfits severely despite ample nominal shot
count.

This is a **NO-GO for the causal Markov decoder**, but it localizes the useful
primitive. Encoding all four events in a round jointly helps more than linear detector
features, whereas generic temporal memory does not. The remaining gap concerns global
error-chain consistency, which matching enforces directly. The next control therefore
constructs an approximate circuit-level detector error model and runs PyMatching;
learned soft components advance only after that graph baseline is reproduced.

Machine-readable results are in
[`results/rigetti_markov_decoder.json`](../results/rigetti_markov_decoder.json).
