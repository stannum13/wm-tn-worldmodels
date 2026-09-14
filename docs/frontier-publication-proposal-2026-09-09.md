# Beyond detection rate: a compiled logical-risk feedback interface

Research proposal, 9 September 2026. This is a prospective research and publication
plan, not a report of a new SOTA decoder. New calculations in this review are the
exact repetition-code diagnostic in Section 3 and a small enumeration check of the
Walsh identity in Section 4. No new surface-code campaign or
hardware-control experiment has been executed for this proposal.

## 1. Recommendation and publication claim

The highest-upside direction is a **small, decision-focused feedback interface
shared by decoding and calibration**. Its job is to preserve information needed to
choose a useful intervention, not reconstruct every physical noise parameter.

The question is:

> Can a compiler extract a small set of circuit- and decoder-informed statistics
> that predicts which graph update or bounded control adjustment will reduce
> logical loss, including under drift that changes the error mechanism?

The immediate executable wedge remains decoder adaptation: repair the time-template
transfer failure and the failed conditional-risk compression. The higher-impact
extension is to use the same typed telemetry to guide physical calibration in an
interactive simulator, then prospective hardware. This is one architecture with
different action backends, not a claim that a decoder-only experiment already
demonstrates a self-calibrating processor.

Candidate paper title: **Beyond Detection Rate: Compiling Logical-Risk Feedback for
Adaptive Quantum Error Correction**.

An ambitious result sentence, conditional on future evidence:

> A bounded symbolic feedback interface preserves most of a rich teacher's
> intervention advantage, transfers across untrained temporal noise profiles, and
> reduces the samples or compute required to maintain logical performance.

Replace “most,” “transfers,” and “reduces” with measured numbers only after the
corresponding gates pass. Do not claim first-ever hybrid decoding, new Bayes theory,
universal sufficient statistics, or quantum-hardware SOTA from simulator results.

## 2. Why this direction now: actual literature hooks

The following sources motivate specific experiments; they do not establish that
our proposed synthesis is unprecedented.

| Primary work | Established result or explicit opening | Consequence for this program |
| --- | --- | --- |
| [Sivak et al., Nature, July 2026](https://www.nature.com/articles/s41586-026-10759-2); [accessible methods/outlook](https://arxiv.org/html/2511.08493v2) | Syndrome detection rates drive physical calibration; the work reports about 20% further logical-error suppression after conventional calibration. Its outlook proposes observation-conditioned policies and learned system models; the preprint also discusses readout/reset extension and the cost of exploration. | Compare a feature-conditioned, bounded controller with detector-rate control using the same optimizer and exploration budget. All exploration shots must count toward performance. |
| [Gong and Hu, August 2026 preprint](https://arxiv.org/html/2608.05686v1) | Local detection-rate convexity and locality-aware optimization are analyzed under specified control-error assumptions. Section IV explicitly proposes model-based controllers and joint fast decoding/slow calibration, and identifies stronger nonconvexity and correlated drift as further work. | Use locality-aware SPSA as a serious control baseline. Test within the favorable local regime as well as cross-channel tradeoffs; do not present an out-of-assumption counterexample as disproving the theorem. |
| [Cao et al., dMLE, February 2026 preprint](https://arxiv.org/html/2602.19722v1) | Differentiable syndrome likelihoods use planar/tensor-network structure. The authors propose cheaper approximations and gate/pulse-level extensions. | Use likelihood estimation as a slow teacher and strong baseline; compare learning the full noise model with preserving only intervention rankings. |
| [Kwak et al., L-NBP, August 2026 preprint](https://arxiv.org/html/2608.27682v1) | BP-derived representations are trained for logical decisions rather than physical-error reconstruction. | A logical objective alone is not our novelty. Test explicit symbolic parity/motif features against learned belief-derived features at matched resources. |
| [Kishi et al., soft outputs, 2026 preprint](https://arxiv.org/html/2602.03336v1) | Bounded and extra-cluster gaps reduce confidence-computation work; the conclusion leaves FPGA realization for future work. | Reuse decoder-generated geometry as telemetry and compare against these confidence signals. Generic confidence gating is already prior art. |
| [Tesseract](https://arxiv.org/html/2503.10988) | Search-based decoding exposes accuracy/work tradeoffs; Appendix B explicitly suggests automating beam-parameter selection. | A bounded, queue-aware program choosing search effort is a legitimate secondary experiment, with the full fixed-beam frontier as its comparator. |
| [Molavi et al., March 2026 preprint](https://arxiv.org/abs/2603.20127) | Symbolic error polynomials and constrained optimization analyze decoder accuracy and robustness. | Use formal analysis for small cases where possible; a robustness certificate or symbolic probability expression is not automatically a new contribution. |

Additional direct comparators are [DGR](https://arxiv.org/abs/2311.16214),
[Neural MWPM](https://arxiv.org/abs/2601.00242v2), and
[QAdapt](https://arxiv.org/html/2607.28422v1). Neural MWPM has an
[author repository](https://github.com/Yotampd/Neural-Minimum-Weight-Perfect-Matching-For-Quantum-Error-Codes).
Reproduce compatible published settings before porting methods to our task. Mark
ports and unavailable implementation details explicitly. QAdapt's reported timing
is backend-only, not complete inference latency.

## 3. Two exact diagnostics: inadequate statistics versus missing information

### A. Fewer detector events can accompany more logical failures

Consider the three-qubit bit-flip repetition code, perfect parity measurements
`s=(e1 XOR e2, e2 XOR e3)`, and the fixed majority decoder. The following is a valid
but deliberately constructed Pauli-mixture channel, not an independently calibrated
physical actuator model:

\[
P(e=010)=a(\theta)=0.02(1-\theta),\quad
P(e=110)=b(\theta)=0.001+0.005\theta,
\]

with identity probability `1-a-b` and `0 <= theta <= 1`.
The first fault is correctable and flips both checks; the second causes majority
decoding to fail and flips only one check. Therefore

\[
C(\theta)=\frac{\mathbb E[s_1+s_2]}2
=a+b/2=0.0205-0.0175\theta,
\qquad L(\theta)=b=0.001+0.005\theta.
\]

| Control parameter | Detector-event rate | Logical error | `E[s1 XOR s2]` |
| --- | ---: | ---: | ---: |
| 0 | 2.05% | 0.10% | 0.10% |
| 0.5 | 1.175% | 0.35% | 0.35% |
| 1 | 0.30% | 0.60% | 0.60% |

Exact enumeration of the three channel outcomes reproduces these values. Here a
single symbolic statistic tracks the relevant risk while the average detector
rate gives the wrong optimization direction. This establishes a diagnostic
possibility, not how often it occurs on hardware. It neither refutes local
convexity results around a common calibrated optimum nor proves that this feature
generalizes. The experiment must include regimes in which detection rate is already
an adequate objective.

### B. Sometimes no syndrome feature can help

In model A, the only nonidentity fault is `100` with probability p; in model B it is
`011` with the same probability. Both give syndrome `10`, but require different
logical corrections. The entire iid syndrome-stream distribution is identical.
Unlimited syndrome history cannot identify the model. A randomized choice between
the two corrections has worst-model logical risk at least p/2; a model-informed
decoder can achieve zero in this restricted channel family.

This is a standard identifiability argument, not a new theorem. It supplies a
mandatory negative control: distinguish **a bad feature map** from **insufficient
observations**. For the second case, analog evidence, a reset herald, a diagnostic
probe, or a justified prior is needed; inventing more latent dimensions cannot
recover information that is absent.

## 4. Concrete architecture

```text
syndrome prefix + available analog/reset metadata + decoder telemetry
                              |
                   typed symbolic feature program
                              |
                  bounded predictive state / uncertainty
                              |
              predicted value of each permitted intervention
                   /                   |                 \
          graph publication     bounded search budget    slow calibration
                   |                   |                 |
          trusted decoder       logical deadline      future QEC circuit
```

Training may use complete trajectories and expensive teachers, but each runtime
feature has a declared availability time, graph support, state cost and meaning.
Completed-shot features cannot be presented as within-cycle observations.

The reduced world model predicts short-horizon changes in logical risk and service
cost under a small action set. It need not simulate a quantum wavefunction. An
action-preserving representation is sufficient only for the declared task, action
set and noise family; this is not universal statistical sufficiency.

### Symbolic feature dictionary

| Feature family | Example | Why include it | Cost caveat |
| --- | --- | --- | --- |
| Circuit-derived parity motifs | XOR of a few detector bits selected from known fault signatures | Distinguish error mechanisms with similar event density | Select motifs on development data; count support/index storage |
| Connected statistics | Local pair covariance; bounded triple coincidence | Separate independent events from a correlated mechanism | More correlations can increase estimator variance |
| Temporal contrast | Prefix/suffix count difference; multiscale Haar sums; run length | Locate a within-record change instead of assuming a midpoint | Suffix access delays availability; do not double-count dependent emissions |
| Decoder geometry | Cluster temporal span, boundary proximity, capped gap | Measure proximity to a logical decision boundary | A gap may cost another solve; charge that cost |
| Innovation | Observed-minus-predicted motif count; lagged residual correlation | Detect when the current state law is inadequate | Predictive calibration does not by itself imply logical improvement |
| Service state | Queue depth, age of oldest unresolved decision, recent service runs | Handle bursts in expensive inference demand | Scheduling effects need measured service traces |

Start with bounded enumeration/forward selection, sparsity penalties and validation
on downstream loss. Compare with equal-budget affine/spline, shallow tree, and small
learned-belief baselines. Do not introduce RL for dictionary search before these
simple searches fail. Hold out geometry/temporal profiles as well as random seeds.

### A logical-contrast teacher, not only a noise-likelihood teacher

For one logical bit, define

\[
J_m(s)=P(s,\ell=0\mid m)-P(s,\ell=1\mid m).
\]

For a pre-syndrome mode belief b, the sign of `sum_m b_m J_m(s)` gives the Bayes
logical decision under the specified model. Compare fitting this contrast or
pairwise action-risk differences with fitting syndrome likelihoods, mode labels,
or teacher latents. Preserve the known distinction between the small history state
and a potentially complicated function of the current syndrome.

For an independent Bernoulli DEM with `s=Hx`, `ell=Lx`, and n detector bits,

\[
J(s)=2^{-n}\sum_{u\in\{0,1\}^n}(-1)^{u\cdot s}
\prod_{j:(H^Tu)_j\oplus L_j=1}(1-2p_j).
\]

This standard Walsh identity generates an explicit parity-feature teacher. Test
whether a sparse or structured subset works; **do not assume sparsity**. Severe
cancellation can make naive largest-coefficient truncation ineffective. Exact
tables remain a small-code diagnostic; structured tensor or motif approximations
are required beyond that regime. Estimate downstream regret, not just coefficient
or likelihood error. The existing conditional-risk head's 11.3% recovery remains
a failed result until a new feature family actually improves it.
The identity was checked against exhaustive enumeration for three independent
Bernoulli faults and two syndrome bits; this is an algebra check, not a scaling
result.

## 5. Ranked experiments and prospective gates

These gates are proposals to freeze with an executable protocol before any new
confirmatory run. They are not replacements for failed gates in archived studies.

### E1. Is the missing ingredient a decision-relevant feature?

Use the exact d3 teacher and new disjoint feature-discovery, fitting, selection and
evaluation data. Compare raw/pair features, parity motifs, temporal contrasts,
decoder geometry, and a small learned representation under matched resource caps.
Test distinguishable equal-density mechanisms and the exact aliasing negative
control above. Retain all harmful overrides as well as rescued failures.

**GO:** at least 80% of the *restricted-action* teacher gap recovered, with a
>=0.02-percentage-point improvement over the matched outcome-trained head
established across ten fresh full fits; the final
80% gate must be supported by a confidence bound, not a favorable ratio alone.
Use <=64 selected statistics and <=1 KiB persistent state per patch as initial
compiler caps, while reporting constants, temporary buffers and graph memory
separately. Evaluate noise regimes with no useful memory as null controls.
On the earlier nominal d3 result, 80% recovery would correspond to about 1.848%
logical error, compared with the conditional-risk student's 1.945%. This is a
planning scale, not a target evaluated by reusing the old test partition.

**Critical ablation:** hold the action library fixed. Otherwise improved risk
prediction and access to new corrections become conflated.

### E2. Compile temporal graph weights compositionally

Replace AA/AB/BA/BB templates with stationary A/B fault primitives plus an inferred
change location. A proposed emission model uses causal temporal contrasts and a
bounded one-change filter. A graph compiler maps inferred schedules to valid fault
probabilities, preserving correlation factors and XOR composition when multiple
independent faults share a signature. It must not average log weights and call
that probability marginalization.

Test known-cut compilation first, then inferred cutpoints. Include the complete
fixed-template catalogue as a baseline where tractable: savings must come from
composition, not withholding candidate graphs from competitors. Compare with
memoryless selection, rate-only adaptation, DGR-style estimation, and compatible
Neural MWPM. Use both logical bases and d5/d7, with unseen timing/noise combinations,
zero changes, two changes and gradual ramps.

**GO:** recover >=70% of the static-to-actual-cut reference gap on declared unseen
one-change tests, supported by adjusted uncertainty, and retain a useful nominal
advantage. The old two-thirds numbers imply a planning target near 1.355%, versus
the current 1.510%; they are not a new test set. Null/shift harm must satisfy the
tighter of +0.02 percentage points and +2% relative to its matched reference, using
the prespecified upper confidence bound. If the oracle gap is too small, report
absolute risk and do not use an unstable recovery ratio.

**Failure interpretation:** a correct known-cut compiler with a poor inferred-cut
policy points to observation/state estimation; failure even with the cut revealed
points to the graph representation or implementation.

### E3. The flagship extension: co-designed calibration and decoding

Build an interactive circuit-noise environment with independently specified bounded
actuators: gate-parameter drift initially, then readout/reset tradeoffs and local
correlated faults. Parameter-response models require physical justification and
held-out alternatives. Start in the local regime favorable to detection-rate
optimization, then introduce the explicitly harder families.

Use a 2x2 factorial: fixed/adaptive physical control x fixed/adaptive decoder.
Compare locality-aware SPSA and the published detector-rate policy recipe with the
same algorithms using our feature-based risk signal. Include dMLE-style estimation
where tractable and a privileged true-risk optimizer as an opportunity reference.
Compare matched sample budgets and optimizer capacity; do not attribute a stronger
optimizer's gain to the feature program.

**GO:** either at least 2x fewer QEC samples to reach a prespecified logical-risk
target, or at least 20% less cumulative logical loss at equal samples, with adjusted
uncertainty and no material stationary harm. Freeze one of these as primary; report
the other as secondary rather than opportunistically taking whichever passes.
Count exploration, probing, waiting and recovery periods, not only the final policy.
For a drift-tracking claim, cumulative loss is the preferred primary endpoint.
Online policies receive neither simulator hidden modes nor true logical labels;
those are reserved for privileged references and independent evaluation. Any
offline teacher-training access is disclosed and matched across learned controls.

Keep the critic and evaluation decoder frozen while testing physical control;
report joint adaptation separately. Otherwise the controller may improve its own
estimated confidence without improving the logical state. Real data can test
features and predictions, but logged decoder-only replay cannot evaluate the
unobserved consequences of new physical actions.

### E4. Optional systems paper: burst-aware value of computation

Use capped logical-gap/motif features plus service-state history to choose bounded
search width or a local stronger solve. Compare with the entire tuned fixed-budget
frontier, random routing, syndrome density, innovation, confidence/cluster-gap
routing, and a controller using queue state without the proposed features.

The distinguishing test is a family with matched average escalation rates but
different temporal clustering of difficult records. Mean load alone can conceal
deadline failures. End-to-end value includes both decoding accuracy and the extra
logical exposure caused by waiting for an actual logical branch. Do not assert that
decoder backlog automatically causes a particular physical error rate.

**GO:** >=2x reduction in deadline misses or p99 response at matched logical
noninferiority, retaining >=80% of the full-teacher net logical gain. Choose the
primary service endpoint and arrival/deadline contract before testing. All routing,
soft-output, interconnect, reset/publication and fallback work counts. Discarding
hard shots is not an accuracy win unless the task explicitly permits postselection
and includes its cost.

Do E4 only if E1/E2 or a strong decoder comparison reveals usable rescue selectivity.
Our earlier entropy routers did not establish it.

## 6. What could become theory, and what is already standard

The existing decision-risk decomposition, the Fourier identity above, and the
aliasing argument are standard tools. They help construct tests but are not a
new-general-theorem claim.

A meaningful theory target is a restricted feature/program class with a verified
error bound over a specified nuisance region. For a fixed syndrome, if each
estimated action risk has an interval containing its true risk, choose an action
without further computation when its upper bound is below every alternative's
lower bound. This is ordinary interval dominance. The research challenge is making
the intervals cheap, nonvacuous and valid under specified drift—not renaming the
decision rule.

Do not transfer fixed-syndrome certificates across new syndrome records. MWPM cost
optimality is not Bayes logical optimality; a cost certificate does not certify the
true physical logical error rate. Finite-particle disagreement is not a coverage
certificate for unmodelled noise. Global stability does not follow from bounded
probability state.

## 7. Evidence and publication ladder

1. **Technical note now:** existing wins/failures, exact feature/observability
   diagnostics, literature map, and prospective interface. No new SOTA claim.
2. **Decoder-architecture paper:** E1/E2 success, external adaptive baselines,
   unseen-geometry/timing confirmation and full resource accounting.
3. **Control-architecture paper:** E3 success in an auditable interactive simulator,
   then independent hardware confirmation if available. Clearly separate the two.
4. **Deployment contribution:** E4 or an end-to-end hardware implementation with
   matched accuracy, resource and workload comparisons.

The next compute budget should increase breadth of falsification, not merely the
shot count under already-selected favorable generators. Freeze model-selection
budgets, split entire noise families and trajectories, use full-fit uncertainty,
and report multiplicity-adjusted primary tests. Never select the strongest-looking
paper claim after looking at all final endpoints.

Reusing the exact teacher, circuit/DEM parser, native frontend and archive verifier
keeps E1/E2 close to the existing implementation. Physical control and live hardware
remain genuinely new work. The recommendation is to do the smallest E1 diagnostic
and E2 compiler check first, then commit the bulk of the next research phase to E3
only if the feedback representation demonstrates real decision value.
