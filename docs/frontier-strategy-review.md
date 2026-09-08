# Frontier strategy review: from world models to adaptive analog decoding

Research decision note, 8 September 2026. This document separates results already
observed in this repository from proposed work and literature anchors.

## Decision

The program now has one architectural thesis:

> **Learn the smallest stable adaptation law around a trusted decoder rather than
> learning the decoder itself.**

Here, “trusted decoder” means a combinatorial or symbolic decision layer that enforces
the code constraints. MWPM exactly solves its weighted-matching objective; it is not
generally a maximum-likelihood decoder for correlated noise. The learned component is
therefore an estimator and adapter around that layer, not a claim to have learned the
code constraints.

The corresponding system is a **three-rate, graph-native analog QEC architecture**:

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

## Three scientific hypotheses and one compiler

### H1 — local sufficient state

Attach a tiny state to a stabilizer or identifiable edge class rather than maintaining
one unconstrained global latent:

\[
z_{v,t+1}=\operatorname{sat}\left(A_vz_{v,t}+g_{v,t}B_v\phi_v(o_t)\right),
\qquad \|A_v\|_2\leq 1-\epsilon,
\]

\[
\Delta w_{e,t}=\delta_{\max}\tanh\!\left(c_e^\top[z_{u,t},z_{v,t},\psi_{e,t}]\right),
\qquad
w_{e,t}=\operatorname{clip}(w_e^{(0)}+\Delta w_{e,t},w_{\min},w_{\max}).
\]

Contraction now applies even when the injection gate is closed. The bounded residual
cannot replace the calibrated structural prior. Relevant inputs are I/Q innovation,
consecutive surprisal, local detector-cluster shape, reset state, and disagreement
between analog and syndrome evidence. The test is whether a few bytes of state per
local element recover the gap to an adaptive teacher on logical error. A predicted
failure boundary is equally important: bounded reweighting cannot create a missing
long-range correlated mechanism.

The first architecture candidate is a **contractive local-global residual filter**,
not a generic RNN. Each site receives a small bank of fixed stable poles (for example,
fast, medium, and slow leaky integrators), while a 2--4-dimensional robustly pooled
global state tracks common-mode drift:

\[
z_{v,t+1}=\operatorname{sat}(\Lambda_vz_{v,t}+B_v\phi_v(o_t)),\qquad
h_{t+1}=\operatorname{sat}(\Lambda_hh_t+B_h\operatorname{robustpool}_v\phi_v(o_t)).
\]

Both diagonal transition matrices have spectral radius below one, and edge residuals
read only their endpoint states plus the tiny broadcast state. This is our proposed
middle point between independent local filters and AlphaQubit-style global mixing. It
borrows the long-memory motivation of state-space world models and the residual
adaptation pattern used in recent robot world-model transfer, but makes stability,
state bytes, spatial access, and the trusted-decoder interface explicit. Retain the
global state only if it closes a prespecified part of the local-to-shared-oracle gap;
otherwise compile it out.

### H2 — finite-grammar sufficiency

A rich teacher's useful adaptive behaviour may lie close to a small fixed grammar:

```text
EMA | CUSUM | clipped integrator | two-state switch | local neighbor sum |
piecewise-linear LUT | bounded edge-class update | atomic plan swap
```

The emitted artifact is fixed-point microcode or a finite-state selector, not a neural
runtime. It may imitate teacher updates during search, but final selection uses logical
error. The principal compression metric is

\[
G_{\rm recovered}=\frac{L_{\rm static}-L_{\rm compiled}}
{L_{\rm static}-L_{\rm teacher}}.
\]

### H3 — ambiguity predicts value of computation

Maintain hypotheses over a tiny nuisance mode—not over the exponentially large Pauli
configuration—and escalate only when their entropy, estimator disagreement,
innovation whiteness, or surprisal runs predict a benefit from expensive inference.
The hypothesis is that these quantities predict the *marginal decoding benefit* better
than syndrome density, raw confidence, largest innovation, or consecutive-event count.
Report logical error versus escalation fraction and failure capture: among rounds that
contribute to a cheap-lane logical failure, how many were flagged in the hardest 1%,
5%, and 10%? Random escalation and each simple score are mandatory controls.

### Hardware-aware decoder program compiler — search mechanism

A reduced world model can help at compile time as a screening surrogate for candidate
programs, graph updates, quantizers, and schedules. It is not the authority for archive
admission: every candidate must be replayed through the trusted decoder or simulator.
Use a Pareto vector rather than a prematurely scalarized reward:

\[
F(P)=(L_{\rm logical},p50,p99,\text{energy},\text{LUT},\text{BRAM},
\text{state bytes},\text{instability},\text{escalation rate}).
\]

Latency deadlines and contraction are hard constraints. MAP-Elites descriptors start
with state bits/site, spatial radius, number of modes, and escalation fraction. Use RL
only when compilation is sequential—for example, placing memory or choosing a probe.

This hypothesis draws on decoder-free latent planning in
[TD-MPC2](https://arxiv.org/abs/2310.16828), real-world latent adaptation in
[DayDreamer](https://arxiv.org/abs/2206.14176), and compact analytic policies from
[symbolic value/policy search](https://arxiv.org/abs/1903.09688). It differs by using
the model as a compiler for a verifiable causal decoder rather than as the deployed
policy.

The memory design is additionally motivated by
[Recall to Imagine](https://openreview.net/forum?id=1vDArHJ68h), which uses structured
state-space memory for long-horizon world-model tasks, and
[ReDRAW](https://proceedings.mlr.press/v331/lanier26a.html), which adapts a pretrained
robot world model through residual corrections in latent dynamics. Neither result is
evidence for QEC; the transferable ideas are stable compressed memory and residual
rather than wholesale replacement.

### Runtime form — finite mode library instead of continuous online planning

Compile a small library of decoder configurations for normal, drift, leakage-like, and
artifact regimes. A stable finite-state selector chooses among them and falls back to
the frozen affine graph under uncertainty. This converts rich offline counterfactual
reasoning into bounded runtime behavior and permits exhaustive transition testing.

## First decomposing benchmark

Use a repetition code or tiny surface code with a known switching nuisance process
over nominal, drift, and burst modes. Compare, on identical trajectories:

```text
static graph -> oracle mode-aware graph -> rich recurrent/HMM teacher
             -> local contractive updater -> compiled EMA/CUSUM/FSM
             -> compiled student plus ambiguity router
```

This yields four separately interpretable gaps: static-to-oracle opportunity,
static-to-teacher learned recovery, teacher-to-student compilation loss, and
cheap-to-routed value of computation. Include a stationary null, a local-mode regime,
and a deliberately nonlocal correlated regime. The latter tests the predicted limit
of bounded local reweighting rather than hiding it.

Advance H1 only if local state recovers at least 80% of teacher gain at no more than
16 bytes/site and does not worsen the stationary-null LER by more than 0.10 percentage
points. Advance H2 only if the compiled grammar recovers at least 80% of teacher gain,
meets the same LER bound, and uses no learned hot-path matrix operation. Advance H3
only if its failure capture at fixed 1%, 5%, and 10% escalation beats every simple
router baseline with a positive paired interval. These thresholds are program choices,
not claims imported from the literature, and will be frozen before the first sweep.

## Evidence and replay boundary

All currently opened stability-8/9 files are development data. Their row order is not
verified wall-clock chronology, and contiguous row blocks are not independent device
acquisitions. A causal adapter on them must use past observations only, reset at
declared trace boundaries, and never tune on logical labels. Confirmation requires
new timestamped acquisition sessions or a never-opened acquisition-level outer holdout;
the session, not an individual shot or row block, is the population-level replication
unit.

Fixed-record replay is valid for decoder-only changes when the decoder produces a
terminal prediction or Pauli-frame update that did not alter later measurements. It
is not a valid counterfactual for physical correction, adaptive reset/measurement,
pulse selection, early stopping, or any other action that changes the future. Those
claims require an interactive backaction-consistent simulator, supported off-policy
data, or prospective randomized hardware.

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
