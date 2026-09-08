# Frontier strategy review: from world models to adaptive analog decoding

Research decision note, 8 September 2026. This document separates results already
observed in this repository from proposed work and literature anchors.

## Decision

The highest-leverage near-term program is a **three-rate, graph-native analog QEC
system**:

1. a deterministic hot path maps raw I/Q to bounded likelihoods and feeds a local or
   matching decoder;
2. a small causal mode estimator slowly updates identifiable edge classes for drift,
   leakage, and readout-state changes;
3. an expensive particle, tensor, or neural teacher runs only when posterior geometry
   shows that the cheap state is inadequate.

This is a world model in the operational sense: it maintains only predictive state
that changes a downstream correction or control decision. It is not an end-to-end
latent simulator, and the current evidence does not justify putting a large recurrent
model in the fast loop.

## Why this is the best-supported direction

| question | strongest observed result | remaining target |
|---|---|---|
| Does analog information help the logical endpoint? | affine soft matching reduces 23-round reset error 16.515% to 16.0625% and 25-round no-reset error 20.415% to 18.2775% | beat released 15.901%/17.623%, or establish <=0.10-point noninferiority at materially lower cost |
| Does more flexible emission modeling help? | spline/KAN improves held-out I/Q Brier 6.0% but worsens logical point estimate to 16.2075% | improvement must mediate logical error, not only calibration loss |
| Does temporal adaptation help? | rolling affine improves next-block Brier 2.49%, but logical error only 0.062% relative with an interval crossing zero | require a mode or mechanism absent from the affine control before adding state |
| When do particles help? | 64 particles recover 91.3% of the EKF-to-grid loss gap for a wrapped multimodal posterior; only 5.5% on an ordinary nonlinear unimodal stream | route by posterior multimodality rather than run particles continuously |
| Is graph structure optional? | independent soft parity, static feature expansions, categorical Markov order, and type-only weights fail or harm transfer | learning should alter bounded graph-native sufficient statistics |
| Is the present CPU path real-time? | fused/preindexed pipeline p50 is 36.95/46.76 us versus 39.1/42.5-us record intervals; p99 is 93.75/128.75 us | stable mean service below cadence on both sessions, bounded backlog, and separately gated tails |
| Can batching or a narrow exact DP rescue it? | batch gain peaks at 19.17% and adds millisecond response delay; exact frontier DP is 50.5/515.9 us and the no-reset width target is structurally impossible | batch-one native fusion, then incremental or hardware-local decoding |

The scientific inference is narrow: on these data the learned component earns its
place as an analog emission model attached to a structural decoder. It does not show
that memory is absent, that KANs or particles are generally ineffective, or that this
system outperforms the released decoder.

## What modern work contributes

The proposed architecture combines ideas that have separately become credible:

- [Pattison et al.](https://arxiv.org/abs/2107.13589) show that soft measurement
  likelihoods can improve MWPM and Union-Find and change the optimal readout-time
  trade-off.
- [Hanisch et al.](https://arxiv.org/abs/2411.16228) find on experimental
  superconducting repetition-code data that eight-bit soft-flip probabilities retain
  full-precision performance in their setting. This motivates an explicit 8/6/4-bit
  sweep here; it does not establish the necessary precision on Ankaa-2 in advance.
- [Caune et al.](https://www.nature.com/articles/s41467-026-73331-6) demonstrate
  sub-microsecond mean decoding per round and 9.6-us logical feedback on Ankaa-2,
  while their best offline result uses soft information and a pairwise graph. Their
  paper explicitly identifies adaptive soft/leakage-aware FPGA decoding as future
  work.
- [Ziad et al.](https://arxiv.org/abs/2411.10343) pair a hardware-local clustering
  engine with a separate adaptivity engine. Under their leakage-dominated simulation,
  adaptivity changes the suppression factor from 2.12 to 3.85 and decoding remains
  below 1 us/round up to the reported sizes.
- [Bausch et al.](https://www.nature.com/articles/s41586-024-08148-8) show that
  per-stabilizer recurrent state, local convolutions, and global mixing can exploit
  leakage, cross-talk, and soft readout on experimental data. This motivates a
  teacher/residual comparator, not an automatic hot-path replacement.
- A recent [Mamba decoder preprint](https://arxiv.org/abs/2510.22724) reports that a
  selective state-space model matches its Transformer comparator on held-out Sycamore
  data with better asymptotic cost. It is promising for the slow causal state lane,
  but its simulated latency-induced-noise model is not a hardware timing result.
- [Neural MWPM](https://arxiv.org/abs/2601.00242) predicts dynamic MWPM edge weights
  with local graph features and global attention. The graph-preserving interface is
  relevant; our hypothesis is that bounded local/state-space residuals can retain most
  of that benefit at substantially lower data and latency cost.

The unoccupied contribution is not any one component. It is an experimentally tested
**information-to-action Pareto curve** joining causal analog calibration, bounded
adaptation, structural decoding, and measured queue behavior.

## Architecture

```text
raw I/Q + hard bits
        |
        v
fixed-point affine or robust emission head       < 2.4 us no-reset budget
        |
        v
bounded edge likelihoods ---> local clustering / persistent graph decoder ---> frame update
        ^                              |
        |                              v
atomic edge-class publication     service/backlog telemetry
        ^
        |
small switching SSM / identifiable EKF          block-rate slow lane
        ^
        |
particle or neural teacher only when mode entropy/miscalibration crosses a gate
```

The affine head remains the default. A robust head may replace it only if it improves
logical error. The slow model predicts graph-native edge-class offsets, not arbitrary
physical gate rates that the observations cannot identify. Publications are atomic
between records. The expensive lane teaches or recalibrates; it never blocks a
deadline.

## New architectural hypotheses

### H1 — graph-local contractive residual memory

Attach a tiny state to a stabilizer or identifiable edge class rather than maintaining
one unconstrained global latent:

\[
z_{v,t+1}=(1-g_{v,t})A z_{v,t}+g_{v,t}B\phi(o_t),\qquad \|A\|<1,
\]

\[
\Delta w_{e,t}=c\tanh\!\left(C[z_{u,t},z_{v,t}]\right).
\]

The contractive transition limits long-horizon state drift; the gate activates memory
when causal innovations indicate a mode change; the bounded residual cannot replace
the calibrated structural prior. Relevant inputs are I/Q innovation, consecutive
surprisal, local detector-cluster shape, reset state, and disagreement between analog
and syndrome evidence. The test is whether this state predicts *logical-error-relevant*
weight changes rather than merely reconstructing observations.

### H2 — posterior-geometry conditional computation

Route to an expensive estimator only when a cheap filter reports high mode entropy,
non-white innovations, or strong estimator disagreement. This turns the wrapped-phase
particle result into an architectural rule: particles are justified by multimodality,
not by nonlinearity alone. The rule must be calibrated to a compute budget and compared
with always-cheap and always-expensive controls.

### H3 — latent-to-symbolic decoder compilation

A reduced world model can be useful entirely at compile time. Train a flexible teacher
to predict downstream logical risk for candidate graph updates, quantization, and
schedules, then distill it into a small grammar:

```text
EMA | CUSUM | clipped integrator | two-state switch | local neighbor sum |
piecewise-linear LUT | bounded edge-class update | atomic plan swap
```

The emitted artifact is fixed-point microcode or a finite-state selector, not a neural
runtime. Search candidates against the multi-objective reward

\[
R=-L_{\rm logical}
-\lambda_1[\bar S-T]_+
-\lambda_2 Q_{0.99}(S)
-\lambda_3 C_{\rm area/power}
-\lambda_4 C_{\rm instability}.
\]

Start with Pareto evolutionary or quality-diversity search because the candidate
language is discrete, evaluation is noisy, and several trade-offs matter. Use RL only
when compilation itself is sequential—for example, deciding where memory should be
placed or which diagnostic probe to request. Every learned surrogate recommendation
must be replayed through the exact decoder before selection.

This hypothesis draws on decoder-free latent planning in
[TD-MPC2](https://arxiv.org/abs/2310.16828), real-world latent adaptation in
[DayDreamer](https://arxiv.org/abs/2206.14176), and compact analytic policies from
[symbolic value/policy search](https://arxiv.org/abs/1903.09688). It differs by using
the model as a compiler for a verifiable causal decoder rather than as the deployed
policy.

### H4 — finite mode library instead of continuous online planning

Compile a small library of decoder configurations for normal, drift, leakage-like, and
artifact regimes. A stable finite-state selector chooses among them and falls back to
the frozen affine graph under uncertainty. This converts rich offline counterfactual
reasoning into bounded runtime behavior and permits exhaustive transition testing.

## Ordered experiments and stop conditions

### F0 — canonical native batch-one path

Compile raw I/Q, hard-bit complementation, odd-parity likelihood composition, cached
edge writes, and decoding into one transaction. Use the C++ result as the canonical
floating/discretization order.

GO requires zero integer-edge-weight, solution-weight, prediction, and restoration
differences on all 80,000 records and adversarial half-integer cases. On five isolated,
counterbalanced passes, the upper 95% block-bootstrap bound on mean service must be
below 39.1/42.5 us. Report p99 service and Lindley-recursion response, maximum backlog,
and drain length. If no-reset median is still at least 42.5 us or the mean interval
crosses cadence, stop wrapper optimization.

### F1 — robust emission without a larger hot loop

Compare affine logistic confidence with a clipped contaminated-Gaussian or small
Student-t mixture and a piecewise-linear lookup-table distillation. Select complexity
only inside calibration data; score logical error on acquisition blocks not used for
selection. Include KAN as the already-tested flexible comparator.

Continue only if the lower paired confidence bound supports at least 1% relative
logical improvement over affine soft matching in both reset regimes, with less than
10% native hot-path latency increase. Calibration-only improvement is a NO-GO.

### F2 — mechanism-gated causal state

Use a two-to-four-mode switching state estimator for normal, drift, leakage-like, and
artifact regimes. Update only identifiable graph-edge classes. Compare no adaptation,
rolling affine, robust EKF, switching SSM, and always-on particles under synthetic
injections whose ground truth is known, then on acquisition-ordered real blocks.

The estimator advances only if it beats frozen affine logical error by at least 1%
relative in both sessions with a positive paired interval, recovers from an injected
change within a prespecified number of records, and does not materially harm stationary
blocks. If predictive calibration gains again fail logical mediation, stop this branch.

### F3 — hardware-shaped adaptive decoding

Port the surviving likelihood/update contract to a software model of local clustering
or another hardware-local decoder. Compare hard local clustering, adaptive local
clustering, hard MWPM, affine-soft MWPM, and a high-capacity neural teacher under the
same observations.

Simulation GO requires soft adaptive local decoding within 0.10 percentage points of
soft MWPM across the real replay endpoints, a credible synthesized path below 1
us/round, and resource scaling reported. This is not a live-hardware claim. Prospective
hardware timestamps and conditional action are required for deployment confirmation.

### F4 — separate world-model control proof

QEC replay cannot establish policy value because actions do not change the recorded
future. Build an interactive stochastic-master-equation benchmark with measurement
backaction, delay, actuator limits, jumps, and artifacts. Compare known-model Bayesian
filtering/MPC, robust filtering, direct recurrent policy, and physics-residual belief
model plus MPC.

World-model GO requires a lower confidence bound above 20% reduction in achieved
infidelity at matched observations and query budget, p99 response below 10% of the
useful intervention window, and no more than 5% degradation in prespecified tail risk.
This is the proper route to a policy claim.

## Portfolio allocation

Allocate approximately 55% of the next compute and implementation budget to F0–F3,
30% to F4, and 15% to intervention design for identifying memory. Do not spend more on
generic KAN, Transformer, particle-count, or batch-size sweeps unless a measured
failure mechanism activates them.

The near-term paper is the analog-to-graph accuracy/latency frontier and its bounded
negative results. The higher-upside paper is the delayed, backaction-consistent belief
control comparison. A strong result in either needs fresh acquisition or prospective
hardware before being described as deployment or SOTA.
