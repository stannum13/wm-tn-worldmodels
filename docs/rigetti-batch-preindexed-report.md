# Transactional batch preindexing: equivalent here, but not a real-time rescue

## Question

Can a C++ batch loop make graph-preindexed matching fast enough by writing each
record's weights into cached graph slots and restoring the base graph only once per
block?

The tested `decode_batch_preindexed(S, W)` path preserves the same discretized MWPM
objective as the endpoint API. For a block of size \(B\) and \(E\) dynamic edges, it
reduces restoration work from approximately \(BE\) edge restores to \(E\), while
keeping endpoint lookup outside the repeated path. This is a throughput optimization;
it necessarily makes the oldest record wait for the remaining \(B-1\) records.

The measured backend is pinned at
[`stannum13/PyMatching@5e0e547`](https://github.com/stannum13/PyMatching/commit/5e0e5472d082f204c2d1c88c51a2a3cca157144b).
The benchmark code is repository commit `5903748`.
An independent API audit then found that an empty batch bypassed plan and width
validation. This did not affect the nonempty campaign, but it was repaired at
[`edf4ea3`](https://github.com/stannum13/PyMatching/commit/edf4ea3) with absent,
stale, wrong-width, decoder-exception, and exact returned-weight tests. The full local
matching suite reports 167 passes and 24 optional skips after that repair.

## Protocol

- Fit the graph and affine I/Q head on the first 60,000 acquisition-ordered records.
- Generate all fused I/Q-to-edge weights outside the timed matching calls.
- Decode the designated final 40,000 records from each Ankaa-2 session through both
  the endpoint batch API and the transactional preindexed API.
- Counterbalance endpoint-first/preindexed-first order by block.
- Test batch sizes 1, 2, 4, 8, 16, 32, and 64.
- Compare every logical prediction and returned solution weight, then verify ordinary
  decoding again after the complete pass.

The predeclared GO condition was zero disagreements plus either at least 20% median
per-record improvement or a cadence crossing **on both sessions**. Front-end work was
excluded from timed calls, so a matching-only cadence crossing is only a necessary,
not sufficient, deployment condition.

## Results

| session | batch | endpoint p50 / record | preindexed p50 / record | reduction | oldest-record p50 | cadence |
|---|---:|---:|---:|---:|---:|---:|
| with resets | 1 | 38.54 us | **31.27 us** | 18.86% | 31.27 us | 39.10 us |
| with resets | 2 | 65.28 us | **52.77 us** | 19.17% | 144.64 us | 39.10 us |
| with resets | 8 | 41.73 us | **35.93 us** | 13.90% | 561.16 us | 39.10 us |
| with resets | 64 | 55.78 us | **48.77 us** | 12.58% | 5,584.54 us | 39.10 us |
| without resets | 1 | 63.60 us | **54.17 us** | 14.83% | 54.17 us | 42.50 us |
| without resets | 2 | 58.12 us | **50.21 us** | 13.60% | 142.92 us | 42.50 us |
| without resets | 32 | 48.11 us | **42.494 us** | 11.67% | 2,677.30 us | 42.50 us |
| without resets | 64 | 48.30 us | **42.63 us** | 11.75% | 5,405.68 us | 42.50 us |

All seven batch sizes preserved every tested prediction and returned solution weight
across both 40,000-record partitions. There are zero
ordinary-baseline disagreements after either session. The focused backend suite has
86 passing tests on GCP, including restoration after an invalid later batch row.

The apparent no-reset crossing at batch 32 is only 6.4 ns per record—0.015% of the
42.5-us interval—and excludes the roughly 7.2-us fused front end measured in the
preceding separate benchmark. Its oldest record waits about 2.68 ms at median. It
therefore cannot be interpreted as an end-to-end cadence crossing. Larger blocks do
not show monotone throughput gains on this shared CPU, and every block above one
sharply worsens response latency. At every tested size, p95 batch service exceeds the
time needed for that batch's records to arrive; queue stability was not demonstrated.

**Predeclared NO-GO.** The largest observed relative reduction is 19.17%, below 20%,
and neither session jointly crosses end-to-end cadence. Transactional batching is an
empirically equivalent API on these partitions and a useful offline-throughput
primitive, but it is rejected as the primary real-time architecture.

## Scientific interpretation

This result localizes the bottleneck. Repeated endpoint lookup and graph restoration
are real costs, but this transactional implementation's measured removal and
amortization did not supply the remaining deployment margin. The negative result also
prevents an easy category error: amortized records/second is not closed-loop
responsiveness. A controller acts on the oldest available observation, so queue-fill
time belongs in the causal latency budget.

The next experiment should remove the Python boundary and the intermediate weight
matrix at batch size one: pass raw I/Q and hard decisions to a compiled affine
likelihood-and-matching path with a persistent graph. Its predeclared gate should be
zero differential disagreements and an upper 95% block-bootstrap confidence bound on
mean service time below the completed-record interval on **both** sessions. A paced
replay should separately report p99 service, Lindley-recursion response latency,
maximum backlog, and drain length. If median no-reset service still exceeds cadence or
the mean confidence interval crosses it, stop wrapper-level optimization. If that
fails, the surviving directions require a genuinely incremental decoder or
hardware-local implementation, not a larger software batch.

Machine-readable results are
`results/rigetti_batch_preindexed_with_resets.json` and
`results/rigetti_batch_preindexed_no_resets.json`.
