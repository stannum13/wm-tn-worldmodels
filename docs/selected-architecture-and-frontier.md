# Selected architecture and quantitative frontier

## What survived

The selected architecture is not an end-to-end learned decoder. It is a structured
minimum-weight matching decoder whose circuit graph stays fixed, with a small affine
head converting each qubit's I/Q point into confidence that the recorded hard decision
is wrong. Only measurement-edge weights change per shot.

For measurement (m), the affine head estimates

\[
q_m(z)=P(1-\hat z_m\mid z_m),\qquad z_m=(I_m,Q_m).
\]

For an edge with non-measurement residual probability (p_e^{\mathrm{res}}), the
shot-specific probability combines independent odd-parity mechanisms:

\[
p_e(z)=p_e^{\mathrm{res}}+q_m(z)-2p_e^{\mathrm{res}}q_m(z),
\qquad
w_e(z)=\log\frac{1-p_e(z)}{p_e(z)}.
\]

The graph structure and logical fault masks come from the embedded Stim circuit. Base
edge probabilities use the Spitz pairwise inversion on calibration syndromes. This
separation is scientifically useful: graph consistency handles global error chains;
the learned component handles only analog emission uncertainty.

## Quantitative position

| Acquisition and endpoint | Strongest hard control | Selected affine soft | Released soft pairwise | Status |
|---|---:|---:|---:|---|
| with resets, 23 rounds | 16.515% | **16.0625%** | 15.901% | mechanism GO; within 0.162 pp |
| without resets, 25 rounds | 20.415% | **18.2775%** | 17.623% | transfer GO; 0.655 pp gap |

The with-resets improvement is 2.74% relative with paired interval +0.048 to +0.857
percentage points. The post-repair no-reset improvement is 10.47% relative with
interval +1.770 to +2.505 points. These beat access-matched hard controls but do not
beat the released decoder.

The remaining accuracy target is below 17.623% on the no-reset 25-round endpoint,
with a paired interval supporting improvement over the released method or an explicit
equivalence margin paired with substantially lower deployment cost. Prepared-state
calibration and the authors' exact graph regularization are unavailable, so exact
same-method reproduction is not currently identifiable from public artifacts.

## What did not survive

- Independent marginal soft parity did not improve stability-9 logical replay.
- Generic static pair and cumulative-parity features harmed or failed.
- Higher categorical Markov order harmed error.
- Three global gate-type weights failed independent transfer.
- A rolling pairwise graph failed the deepest endpoint.
- The spline/KAN I/Q head was worse than affine at the selected interface.
- Rolling affine calibration improved next-block Brier/NLL but produced only 0.062%
  relative logical improvement, with an interval crossing zero.

These results reject specific mechanisms. They do not establish that temporal memory,
KANs, or particles are universally useless.

## Identifiable slow state

If future data restore logical headroom for adaptation, estimate graph-native edge
classes rather than nominal physical gate rates. With detector bits (d), spins
(z_v=(-1)^{d_v}), independent edge faults, and selected parity probes (S):

\[
-\log E\left[\prod_{v\in S} z_v\right] = A r,
\qquad r_e=-\log(1-2p_e).
\]

For a low-dimensional basis (r=r_0+H\beta), only components for which (AH) has
full column rank are identifiable. Rank-deficient edge classes must be merged or
probed separately. A robust ridge or EKF may update (\beta) after completed blocks
and atomically publish a new graph. Logical labels remain outside calibration.

Particles are admitted only after a switching or continuous EKF fails a locked
next-block test and the posterior is demonstrably multimodal. They remain entirely in
the graph-publication lane; PyMatching owns the decision path.

## Deployment boundary

The current Python reference rebuilds 268--plus-edge graphs per shot. P50/p99 are
2.20/2.90 ms with resets and 2.97/3.86 ms without resets. Fixed-graph batch-one decode
is tens of microseconds, so graph reconstruction—not affine inference—is the dominant
cost. PyMatching 2.4 exposes edge addition but no public in-place weight setter.

The next implementation must therefore provide one of:

1. a C++/Rust matching backend with fixed topology and mutable weight buffers;
2. an incremental decoder that consumes per-measurement likelihoods directly; or
3. a compiled local-clustering/FPGA path with the affine head fused into edge loading.

A calibration-only uncertainty trigger was also tested. Routing 20--24% of shots to
the slow decoder recovers only 21--34% of the full accuracy gain; routing roughly half
recovers about 70%. This is not selective enough to avoid backend work. See the
[event-routing report](event-triggered-soft-routing-report.md).

Deployment GO requires stable streaming at 1.7 microseconds per syndrome round, a
stretch p99 below 1 microsecond per round, and separately measured terminal response.
GCP CPU timings cannot be compared directly with the paper's FPGA measurements.

## Novel contribution boundary

The strongest defensible contribution is an **auditable analog-to-graph interface**:
cross-fitted without logical targets, validated across reset regimes, explicit about
which uncertainty is learned, and equipped with a circuit-level test preventing
classical readout error from being confused with persistent quantum state error. A
larger world model becomes scientifically meaningful only if it improves the logical
or control endpoint beyond this interface under matched latency and observation cost.

Primary anchors are [Caune et al.](https://www.nature.com/articles/s41467-026-73331-6),
[Spitz et al.](https://arxiv.org/abs/1712.02360),
[Chen et al.](https://arxiv.org/abs/2110.04285), and
[Pattison et al.](https://arxiv.org/abs/2107.13589).
