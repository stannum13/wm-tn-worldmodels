# Fixed-topology mutable analog decoder

## Controlled question

Can the selected affine-I/Q matching decoder preserve its decisions while replacing
per-record graph reconstruction with fixed-topology weight buffers? This tests an
implementation bottleneck without changing the fitted graph, I/Q model, held-out
records, or MWPM objective.

Upstream PyMatching 2.4 does not expose mutable decode-time weights; its
[issue #172](https://github.com/oscarhiggott/PyMatching/issues/172) remains open.
The experiment therefore pins the unreviewed
[Allenator/PyMatching fork](https://github.com/Allenator/PyMatching/commit/435dc7ec85c10314c09f069a3d924d3a3dee8251)
at full commit `435dc7ec85c10314c09f069a3d924d3a3dee8251`. The runner rejects a backend whose
`decode` signature does not explicitly declare `edge_reweights`.

## Architecture

The graph topology, boundary flags, and logical fault masks are built once. For edge
`e`, residual probability `p_res`, and its mapped measurement surrogates `q_m`, the
packed kernel computes

\[
\alpha_e=(1-2p_e^{res})\prod_{m\in M_e}(1-2q_m),\quad
p_e=(1-\alpha_e)/2,\quad
w_e=\log((1-p_e)/p_e).
\]

One packed affine kernel emits every `q_m`; one indexed kernel emits only the 96 or
104 matched measurement-bearing edge weights. The backend receives endpoint/weight
triples and restores the original weights after each decode.

The original maximum LLR is `9.210240`; 13,427 reset and 32,638 no-reset holdout
records emit at least one larger weight, which otherwise forces the fork to
reconstruct internal state. A new degree-one detector with a
single boundary edge at the clipped maximum LLR `11.512915`, zero syndrome, and empty
fault mask seeds that maximum. Its parity equation forces the dummy edge to zero, so
it is inert in the ideal matching objective. The runner asserts its structure,
endpoint uniqueness, zero syndrome, nonnegative bounded updates, restoration, and
repeatability. This workaround is specific to the audited fork; normalization and
tie behavior are not claimed invariant across arbitrary implementations.

## Differential accuracy result

| Session | Held-out records compared | Prediction disagreements | Logical error | Restore/repeat failures |
|---|---:|---:|---:|---:|
| with resets, 23 rounds | 40,000 | **0** | 16.0625% | 0 |
| without resets, 25 rounds | 40,000 | **0** | 18.2775% | 0 |

The packed I/Q kernel matches the generic implementation exactly at float32. Packed
versus generic edge weights differ by at most `1.86e-12`. Thus all 80,000 real replay
predictions match fresh graph reconstruction, not merely its aggregate error rate.

## GCP CPU timing

| Session | Rebuild+decode p50 | Mutable matching amortized | Speedup | Packed I/Q amortized | Packed weights amortized | Component sum / record | Component sum / round | Packed batch-one compute p50/p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| with resets | 2,170.7 us | 34.80 us | 62.4x | 2.89 us | 3.56 us | 41.25 us | 1.79 us | 95.9/195.2 us |
| without resets | 2,970.0 us | 69.80 us | 42.5x | 2.94 us | 3.87 us | 76.61 us | 3.06 us | 198.3/768.0 us |

Mutable throughput is the median of 30 full 40,000-record calls. The component sum
is not a measured end-to-end service latency: it excludes acquisition, transfer,
queueing, syndrome conversion, and orchestration. The batch-one figure measures the
packed I/Q-to-weight-to-decode computation for 2,000 records, but still runs through
Python on a shared GCP CPU. The 30-repeat tail summaries are descriptive, not a
production tail-latency certification.

## Streaming microbatch Pareto

Could small microbatches reach the 1.7-microsecond-per-round throughput target without
creating an unacceptable response delay? The packed computation was replayed over
4,096 records at batch sizes 1--64. Every microbatch prediction matched the full-batch
mutable result.

| Session | Batch | Median compute / record | Median batch compute | Worst fill wait at cadence-derived record arrivals |
|---|---:|---:|---:|---:|
| with resets | 1 | 108.7 us | 108.7 us | 0.0 us |
| with resets | 4 | 61.8 us | 247.1 us | 117.3 us |
| with resets | 16 | 45.3 us | 725.2 us | 586.5 us |
| with resets | 64 | 40.3 us | 2,577.2 us | 2,463.3 us |
| without resets | 1 | 121.6 us | 121.6 us | 0.0 us |
| without resets | 4 | 74.0 us | 295.9 us | 127.5 us |
| without resets | 16 | 57.6 us | 920.9 us | 637.5 us |
| without resets | 64 | 53.9 us | 3,448.6 us | 2,677.5 us |

The asymptotes remain about 40 and 54 microseconds per record, versus 39.1 and 42.5
microseconds for 23/25 rounds arriving every 1.7 microseconds. Microbatching therefore
does not meet the throughput target, and its millisecond-scale batch completion makes
response worse. At batch 64, oldest-record completion is about 5.04/6.13 ms after
including fill. It is a **NO-GO as the primary latency fix**; the next gain must remove
Python/backend parsing and allocation rather than accumulate more records.

## Decision

**Engineering GO.** Both sessions pass the predeclared mutable-matching gate of less
than 100 microseconds amortized per record and at least 20x speedup, with exact
full-holdout prediction agreement. Graph reconstruction is no longer a necessary
part of this architecture.

**Deployment NO-GO on the present Python path.** Batch-one p99 is 195--768
microseconds per completed record, and even amortized component throughput is
1.79--3.06 microseconds per syndrome round. This does not establish the paper's
1.7-microsecond streaming cadence, FPGA performance, or live-hardware response.

The next smallest contribution is a compiled indexed API that accepts aligned I/Q
and hard bits, evaluates the 24 affine parameters and exact odd-parity weights, and
calls matching without Python endpoint triples or allocations. If that still misses
the deadline, measure the extracted graph's temporal pathwidth before attempting an
exact narrow-strip dynamic program.

Machine-readable results are in `results/rigetti_mutable_matching_with_resets.json`
and `results/rigetti_mutable_matching_no_resets.json`. The microbatch curves are in
`results/rigetti_mutable_microbatch_with_resets.json` and
`results/rigetti_mutable_microbatch_no_resets.json`.
