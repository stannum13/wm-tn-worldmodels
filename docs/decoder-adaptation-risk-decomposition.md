# Small state is not the same as a small decoder

8 September 2026. A decision-theoretic synthesis of the completed campaigns, not
a claim of a new general theorem or a literature-surpassing decoder.

## Why this remains a useful research direction

Adaptive decoding graphs are not new: DGR estimates edge and edge-pair statistics
from decoded matchings to address drift and correlations. It is a relevant future
adaptive comparator, not something our static-grid victories have already beaten.
[DGR, Wang et al., 2024 version](https://arxiv.org/html/2311.16214v3).

Small hidden-state filters also have a concrete physical precedent. Varbanov et al.
used syndrome signatures and analog readout to infer transmon leakage in
simulations; their demonstrated logical-performance intervention was postselection.
That supports the plausibility of a meaningful nuisance state, not our synthetic
four-mode model as a calibrated model of actual leakage.
[Leakage detection for a transmon-based surface code](https://www.nature.com/articles/s41534-020-00330-w).

Neural latency is not a sufficient objection to a neural decoder: AlphaQubit 2's
March 2026 preprint revision reports sub-microsecond-per-cycle decoding through
surface-code distance 11 and color-code distance 9 on commercial accelerators.
Our possible contribution must be a measured accuracy/resource/stability trade-off
under matched conditions, not comparison of unrelated throughput numbers.
[AlphaQubit 2, version 2](https://arxiv.org/abs/2512.07737v2).

The distinctive experimental question is therefore: **which part of adaptation is
small, and which part is not?** Our exact-teacher diagnostic and failed student
compression make that question sharper than a generic neural-architecture sweep.

## Three distinct sources of decision error

For a completed syndrome record s and past observations h, let

\[
q_m=P(m\mid s,h),\qquad
r_{m,a}(s)=P(\text{logical failure of action }a\mid m,s).
\]

Assume the mode m is sufficient for current logical uncertainty once s is known:
logical outcome is conditionally independent of history given (m,s). If not,
redefine the true risks as r(m,a,s,h); the approximation error below must then
include replacing those history-dependent risks with a history-free program. Decoder
action a is a fixed mapping from syndrome to a code-compatible decision. Its risk
is \(L(a)=\sum_m q_m r_{m,a}(s)\).

The controller estimates q by \(\hat q\), approximates r with a small program
\(\hat r\), and chooses \(\hat a=\arg\min_{a\in\mathcal A}\hat q^\top\hat r_a\).
Let \(a^*_{\mathcal A}\) be the best action in its finite graph library and
\(a^*\) the unrestricted Bayes decision. Then, exactly,

\[
L(\hat a)-L(a^*)=
\underbrace{L(a^*_{\mathcal A})-L(a^*)}_{\text{missing action opportunity}}
+\underbrace{L(\hat a)-L(a^*_{\mathcal A})}_{\text{selection regret}}.
\]

At a fixed (s,h), if \(\epsilon_r=\max_{m,a}|r_{m,a}(s)-\hat r_{m,a}(s)|\),
with losses in [0,1], normalized beliefs, and TV defined as half the L1 distance,
the standard approximate-risk argument gives the pointwise bound

\[
L(\hat a)-L(a^*)\le
\text{missing action opportunity}
+2\,\mathrm{TV}(q,\hat q)+2\epsilon_r.
\]

To see this, every true/estimated action risk differs by at most
\(\mathrm{TV}(q,\hat q)+\epsilon_r\); apply this once to the selected action and
once to the true best library action. Estimated optimality cancels the middle
term. This is a loose diagnostic bound, not an empirical certificate: the true
conditional risks and mode posterior are generally unknown on hardware. An
averaged bound averages the pointwise terms; no independence between the two
estimation errors is assumed.

These terms suggest different interventions:

| Source | Appropriate intervention | What does not establish success |
| --- | --- | --- |
| Poor state posterior | Better observations, rate adaptation, localized temporal state | Mode accuracy without logical benefit |
| Poor conditional action risk | Syndrome-conditioned risk features, teacher targets, bounded nonlinear rules | Imitating teacher latents alone |
| Missing actions in the graph library | New time-resolved potentials, correlation factors, local solves or hypothesis width | A more accurate router over the same inadequate actions |

The first term is about **available decisions**, not necessarily missing graph
topology. A temporal weight profile can be missing while every physical fault
mechanism is already represented in the original unrolled graph.

## Why a global mode/action loss table is only an approximation

The mode posterior is conditioned on the same syndrome used by the decoder.
Substituting \(P(\text{failure}\mid m)\) for
\(P(\text{failure}\mid m,s)\) silently discards that dependence. Accurate mode
classification does not make this substitution correct. The time-template
campaign consequently tests global and count-conditioned tables as restricted
programs and reports predicted versus observed risk by difficulty/confidence bin.
It does not call either table Bayes-optimal.

For a fixed syndrome and two noise modes, an exact binary logical decision has
at most one switching boundary in the **pre-syndrome** mode belief b:

\[
(1-b)\,[P(s,1\mid0)-P(s,0\mid0)]
+b\,[P(s,1\mid1)-P(s,0\mid1)]>0.
\]

That says the dependence on history can be extremely small. It says nothing about
the complexity of the two syndrome-dependent coefficients. With more modes,
finite-action decision regions remain intersections of half-spaces in the
mode-belief simplex for fixed s, not generally in raw observations. The bridge
between the pre-syndrome belief b and post-syndrome q is
\(q_m=b_m P(s\mid m)/\sum_j b_j P(s\mid j)\); multiplying by the common positive
denominator gives the affine joint-probability boundary. State estimation
from history and syndrome-dependent action-risk computation are separate
complexity sources, and either can remain expensive.
This explains why a scalar sufficient history state and poor tiny-head teacher
compression can coexist without contradiction.

## What the actual measurements distinguish

The [parent challenge](surface-frontier-challenge-report.md) establishes a nominal
temporal benefit beyond matched memoryless routing, yet fixed persistence causes
independent-mode harm. The [rate-bank follow-up](surface-rate-adaptation-report.md)
repairs most of that harm with eight probabilities; it does not repair noise that
changes within the record.

The [exact d3 teacher](surface-exact-teacher-report.md) improves over the compiled
policy by 0.19531 percentage points on its fresh nominal test. Exact memoryless
inference already accounts for 0.14648 points of that difference, while adding
exact causal history supplies another 0.04883 points. These are within-test
comparisons, not a universal additive attribution across architectures.

The [conditional-risk student](surface-teacher-distillation-report.md) recovers
only 11.32% of the remaining restricted-teacher opportunity. Its mean LER is
1.94481%, versus 1.81931% for the exact teacher constrained to the same two graph
outputs and 1.77589% for the unrestricted exact teacher. Improving the target
slightly helped the action rule; it did not make the small feature/program class
sufficient. The two distinct remaining gaps are about selection and action access.

The [new frozen time-template protocol](surface-time-template-protocol.md) tests
whether resolving a record into early/late noise templates addresses a different
action-library limitation. Its final result must not be presumed from these
earlier observations.

## Stability and deployment are additional conditions

Probability normalization keeps the HMM state bounded. It is not a certificate of
robustness to an incorrect noise model or a common contraction bound for every
implementation. The binary exact-model log-odds update has the contraction described
in the [teacher theory note](surface-exact-teacher-theory.md), but that certificate
must not be transferred automatically to a richer switching-rate bank.

Likewise, selecting one of four graphs before decoding uses one graph invocation,
not necessarily less total storage or one internal matching pass. End-to-end
service time, queueing, graph memory, acquisition timing and fault-tolerant system
integration remain separate empirical requirements.
