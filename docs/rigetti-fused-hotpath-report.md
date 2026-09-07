# Fused affine-I/Q front end: computational GO

## Question

Can the selected small analog mechanism be made cheap enough for streaming by fusing
affine I/Q scoring, hard-bit-conditioned surprisal, odd-parity edge composition, and
LLR conversion into one compiled kernel?

The comparison holds heads, graph, probabilities, and mutable matching fixed. It
changes only computation and allocation. The 2,000 records per session are from the
previously accessed designated test partition, so this is a systems differential,
not a new accuracy confirmation.

## Result

| session | composed front end p50 / p99 | fused front end p50 / p99 | composed pipeline p50 | fused pipeline p50 | record cadence |
|---|---:|---:|---:|---:|---:|
| with resets | 51.17 / 91.20 us | **7.19 / 17.37 us** | 106.01 us | **51.05 us** | 39.1 us |
| without resets | 49.87 / 79.28 us | **7.25 / 21.75 us** | 116.87 us | **60.70 us** | 42.5 us |

Every endpoint/weight value is exactly equal to the composed NumPy path
(`maximum_edge_array_difference = 0`) and there are zero logical-prediction
disagreements in either screen. The front end is 7.1/6.9 times faster at the median.
The contiguous pipeline median falls by 51.8% with resets and 48.1% without resets.
Pipeline and matching-only passes are separate, and AB/BA order alternates by record.

**Computational GO:** this exceeds the predeclared 2x front-end gate without changing
the numerical or decoding result.

**Deployment remains NO-GO:** median pipeline time remains 12.0/18.2 us above completed-
record cadence, and p99 is 109/155 us. Shared-GCP-CPU tails are not FPGA claims, but
they rule out declaring real-time completion from median throughput alone.

## What remains and the next gate

Direct counterbalanced matching-call medians are 35.01/46.37 us. Holding the fused
front-end medians fixed gives component allowances of 31.91/35.25 us under cadence,
requiring at least 8.8%/24.0% matching-call reductions before wrapper and tail effects.
The current backend
reparses 96/104 endpoint pairs and rediscovers graph slots on every record.

The next implementation is an opaque, graph-owned preindexed Tier-1 plan accepting a
contiguous weight vector. It must hard-fail on graph mutation, wrong ownership,
invalid weights, or normalization escape and restore exact base slots on success and
exceptions. Continue only with zero endpoint-API disagreements and at least a 20%
matching-call median reduction; cadence crossing is the stronger GO.

Machine-readable artifacts are
`results/rigetti_fused_frontend_with_resets.json` and
`results/rigetti_fused_frontend_no_resets.json`.
