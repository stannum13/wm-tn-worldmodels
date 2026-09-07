# Real-data and deployment benchmark ladder

The next campaign separates three claims: real-measurement replay, measured inference
latency on our processor, and live physical feedback. Public data plus GCP can support
the first two. They cannot reproduce FPGA or quantum-hardware feedback latency.

| Priority | Benchmark | Public data | Reported anchor | Our valid first claim |
|---:|---|---|---|---|
| 1 | Rigetti streaming stability-code decoding | [CC-BY record](https://zenodo.org/records/15364358): 5.6 MB smoke file, 130 MB serious confirmation file, 1.23 GB total | 9.6 μs full response for nine rounds; 6.5 μs decoding plus 3.1 μs communication/control | Real-hardware syndrome replay, logical error and CPU latency/load |
| 2 | Continuous superconducting-qubit I/Q trajectories | [code/data index](https://github.com/qnl/trajectories_lstm) | 40 ns digitization; 1.5M training and 0.5M evaluation traces in the published study | Past-only state estimation and calibration; exclude backward smoothing |
| 3 | Real TLS jump trace | [214 MB CC-BY archive](https://zenodo.org/records/21908500) | One long superconducting-qubit jump record | Blocked-split hazard likelihood and survival calibration |
| 4 | Google RL-QEC drift adaptation | [7.8 GB CC-BY archive](https://zenodo.org/records/18896801) | Reported 3.5× logical-error stability improvement under injected drift | Logged-data drift prediction; policy improvement only if counterfactual support is adequate |
| 5 | Willow surface-code decoding | [public archive](https://zenodo.org/records/13273331) | Real-time latency 63±17 μs; offline neural logical error per cycle 0.269%±0.008% in the published study | Start with the 5.7 GB subset; decoder replay and queue stability |

The immediate applied benchmark is Rigetti because it combines manageable public raw
data, real superconducting hardware, decoder timings, and a live-feedback reference.
The independently downloadable `fast_feedback_raw_data.h5` is sufficient for schema
and I/Q calibration work. Add `stability_9_raw_data.h5` for the smallest serious
logical-error confirmation; the entire 1.23 GB record is not required.
The continuous-I/Q set is the best next denoiser/head test, but its upstream repository
has no explicit software license; reimplement loaders and do not vendor its code.

Primary sources: [Rigetti feedback](https://www.nature.com/articles/s41467-026-73331-6),
[continuous trajectories](https://arxiv.org/abs/1811.12420),
[TLS jumps](https://arxiv.org/abs/2603.11889),
[RL-QEC drift control](https://www.nature.com/articles/s41586-026-10759-2), and
[Willow surface-code decoding](https://www.nature.com/articles/s41586-024-08449-y).

For every replay report p50/p95/p99 latency, throughput, queue growth, missed deadlines,
calibration, and task error. Never compare vectorized GCP timings directly with FPGA
batch-one latency. A live-deployment claim requires prospective hardware access.
