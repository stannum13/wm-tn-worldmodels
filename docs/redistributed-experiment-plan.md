# Redistributed directional experiment plan

Version 1.1, 7 September 2026; extended to streaming observations and response deadlines. Prepared for a subsequent autonomous experiment campaign; the new experiments below have not been executed. Motivation and literature are in the [background note](world-models-quantum-control-background.md). This plan supersedes earlier recommendations that made the RB forecast result a prerequisite for investigating other primitives, or required fitting a frozen Markov residual before fitting memory.

## Objective and scope

Determine when learned predictive models help quantum control or inference, which primitives account for that help, and what their total measurement and computational costs are. Produce reproducible artifacts suitable for explanatory blogs and, where the evidence supports it, research manuscripts.

“Exhaustive directional experiments” means complete coverage of a declared finite matrix, including negative controls, failed fits, and uncertainty. It does not mean exploring all architectures or parameter values. Start with small systems where ground truth and errors are auditable, then expand branches whose experiments are informative. Publication is an outcome contingent on evidence and novelty, not an acceptance criterion for selecting results.

The initial campaign uses local simulations and public data. Additional hardware acquisition is a later deployment stage dependent on actual access and a measurement protocol. Existing terminal RB data do not supply outcomes for new pulses or intermediate measurements. A surrogate trained on those records cannot validate its own counterfactual recommendations.

## Redistribute the effort

The percentages below allocate the next round's active research effort. They are management choices that protect underexplored directions, not estimates of their probabilities of success. Track actual compute separately. Changes occur only at recorded review points; do not quietly transfer an entire branch's allowance to the currently best model.

| Track | Initial allocation | Scientific question | Relationship to the original plan |
|---|---:|---|---|
| R0: Evidence, data, and reference audit | 10% | Are comparisons valid and uncertainties meaningful? | Repairs Experiment A's foundations and supports every track |
| R1: Predictive state and mechanism recovery | 25% | What information and representation are sufficient? | Promotes A/B and H1/H3 independently of RB success |
| R2: Planning and policy value | 25% | Does modeling improve actual decisions at equal cost? | Adds the missing control endpoint to A/B/E |
| R3: Experimental access and probe selection | 20% | Which measurements resolve useful uncertainty? | Adds an independent identification branch to A |
| R4: Physical constraints and adaptive capacity | 15% | Do constraints or variable capacity improve error–cost trade-offs? | Gives C/H2/H5 an independent route to evidence |
| R5: Spatial abstraction and size transfer | 5% | Is there an inexpensive informative test of RG/tensorial transfer? | Retains D/E as bounded feasibility studies, not rejected hypotheses |

Within R1, every applicable family receives a correct implementation, a positive-control test, and the same declared development-data access before ranking. Initial tuning uses at most three recipes per family and three optimization seeds per recipe on designated development instances. Report performance within equal resource caps and, where useful, under predeclared extended caps with convergence diagnostics. Seed counts measure training variability, not experimental replication.

R2 and R3 may begin with known reference dynamics while learned inference is being repaired. R4 does not depend on demonstrating non-Markovianity. R5 is deferred or expanded based on its own feasibility evidence, never on RB losses.

Protect streaming work inside these allocations: 10 percentage points of the total effort come from R2 for response/control experiments, and 5 from R3 for event detection and acquisition timing. This is 15% of the existing 100%, not an additional allocation. Share causal filtering and observation infrastructure with R0/R1.

## R0: Establish the evidence contract

Preserve the existing results as a development ledger. The `idle100` and `idle180` outcomes have been inspected; subsequent changes evaluated there are exploratory. Record the original 46.9% reduction in mean per-bias `idle100` RMSE together with the lack of an active `idle180` gain, unequal optimization effort, and unavailable acquisition-level uncertainty. Do not convert mean per-bias RMSE to pooled MSE without returning to the underlying rows.

For each new experiment, record the mechanism/instance identifier, data source and commit, timing convention, controls, preparation, measurement instrument, observation access, shot or readout-noise model, calibration blocks, partition, model, optimizer, resource cap, and evaluation rule. Save all forecasts and outcomes with stable identifiers. Label whether the model receives true parameters, estimated parameters, or neither.

Fix four observation contracts before training:

| Contract | What the learner receives | Permitted control claims |
|---|---|---|
| O0: Terminal outcomes | Controls/timings plus final measurements from repeated runs | Offline sequence selection and adaptation between runs |
| O1: Ensemble characterization | Outcomes in several bases from separately prepared/replayed experiments | Richer identification, with all extra preparations counted |
| O2: Feedback record | Discrete syndromes, photon counts, or continuous readout, including timestamps and backaction | Causal policies acting during a run on information already received |
| O3: Privileged reference | Exact reduced or joint states and known mechanism | Solver checks, ceilings, and diagnostic ablations only |

O1 is not a nondestructive view of the same evolving specimen. O2 changes the intervention protocol. Comparisons within a contract match access; comparisons between contracts account for the added experimental resources. In particular, a density matrix available to the simulator is not a free observation for the policy.

The current calibrated analog `p0` records cannot automatically be treated as binomial counts. Reconstruct their measurement pipeline or use a defensible observation model. Seek timestamps and calibration groupings, but never infer independent acquisition blocks from sequence length alone.

For streams, also record sample integration intervals, event/acquisition timestamps, arrival timestamps, missingness, detector saturation, measurement strength/efficiency, preprocessing delay, and the time an action takes effect. Event time and data availability time are different. Use only past-available samples for online predictions; future-window smoothing, whole-stream normalization, and random splits of overlapping windows are prohibited in causal claims. Split by full acquisition/trajectory before making windows. Model state must not carry across independent partitions.

Deliverables: an evidence/coverage ledger, versioned manifests, reference-accuracy checks, and a report distinguishing untested hypotheses from failed inference methods.

## R1: A controlled laboratory for predictive state

Begin with 30 process instances: five in each of six mechanism families. Preassign three per family to method development and two to a first evaluation screen: 18 development and 12 screen instances. The small screen maps directions; it cannot establish a population-wide effect. After screen results are inspected, those instances become development evidence and final confirmation requires fresh instances.

| Family | Mechanism | Purpose |
|---|---|---|
| F1 | Coherent error plus ordinary Markovian dissipation | Memory is unnecessary; test false positive claims and simple-model competence |
| F2 | Gate-dependent or duration-dependent Markovian error | Determine whether omitted controls imitate a memory advantage |
| F3 | Quasi-static classical detuning, redrawn under a declared reset protocol | Test persistent classical information within an episode |
| F4 | Finite-correlation classical switching noise | Test memory times and departure from the quasi-static assumption |
| F5 | Small quantum environment with a tunable coupling and dissipation | Test representable retained quantum information without forcing dissipation into a pure reservoir |
| F6 | Leakage into a third level | Test whether an omitted system degree of freedom is misread as environmental memory |

Initially use fixed SPAM. Add SPAM uncertainty, control distortion, and between-run drift as separately labeled stresses. Vary one factor at a time before testing selected combinations. For reset-sensitive families, specify whether the bath/latent variable resets between sequences and across measurement repetitions; do not assume identical repetitions when the environment is carried forward.

Use identical physical controls and timing descriptions for all applicable learners. Compare six candidate families: general/constrained-control-dependent Markov channels; classical hidden-state dynamics; dissipative quantum memory; spectral or linear predictive-state methods; a GRU; and a causal process-MPO. Include the simple RB model where its observable is applicable. Supply action geometry and time to generic models as well as physical models. A token-only versus geometry comparison is a separate ablation.

Spectral and transfer-tensor methods require particular observations; use them where those assumptions hold and report missing support elsewhere. A finite dephased D2 model does not represent every possible classical-memory process. Compare classical capacities explicitly. Retain/reset/dephase tests require both fixed-parameter ablations and retrained alternatives; equal one-step channel capacity matters.

First verify each model on easy realizable data generated independently of its implementation. A provisional recovery tolerance is terminal-probability RMSE ≤0.001 on at least 9 of 10 easy noiseless trials with adequate controls and training data. Failing this check routes work to optimization/implementation diagnosis; it does not refute the family. This tolerance does not constrain deliberately misspecified or limited-access tasks.

For the first predictive screen, use training lengths 2–16 and held-out lengths 24 and 32, with length 64 as a declared stress test. Development validation uses disjoint sequences from the training horizon. Compare two new control families as well as length transfer. An initial data-budget curve uses 256, 1,024, and 4,096 training sequences, with measurement repetitions fixed within each curve. Start with the middle budget, then complete the curve. Freeze the actual control sets, noise strengths, and fit caps after solver/feasibility pilots and before screen outcomes are accessed.

Fit all models to the observed outcomes; a simple model's residual is a diagnostic or initialization, not unquestioned ground truth. Report training error, test error, within-length structure, physical validity, optimized-control prediction error, model size, and runtime. Track the number of optimizer failures.

Add the 1,000-trajectory QD3SET-1 spin-boson benchmark after the small-mechanism checks. Start with training through \(t\Delta=10\), evaluation through 20, and a declared bath-parameter holdout. Population and coherence forecasts are separate endpoints. New interventions and two-time responses need additional converged HEOM/process-tensor data; do not derive them from unperturbed trajectories without justification. [QD3SET-1](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2023.1223973/full), [OQuPy](https://arxiv.org/abs/2406.16650)

Deliverable: a mechanism × access × model performance map identifying information limitations, training failures, and usable predictive advantages.

## R2: Put the models to work as controllers

Start with two tasks: preserve an unknown input state through a fixed noisy interval, and implement a prescribed single-qubit rotation. Evaluate preservation and gate behavior over the six Pauli eigenstates, or an explicitly richer test ensemble. A controller that simply resets every input to one favorable state must fail this test. Count all measurement preparations needed to estimate this objective.

For a small first comparison, define a common finite catalogue of feasible control sequences containing standard echo/XY sequences, simple calibrated pulses, and prespecified alternatives. The known-dynamics reference can evaluate every catalogue entry, providing the exact best value **within that catalogue**. This gives an unambiguous selected-action regret. Continuous-pulse optimization is a later extension; a GRAPE result is then a numerical reference, not a certified global optimum.

All controllers share pulse amplitude, bandwidth, duration, observation access, initialization, and verification budgets. Parameter values are hidden except from explicitly privileged references. Compare:

1. Fixed robust sequences and simple calibration.
2. Direct experimental-style search: allocation/bandit methods for the finite catalogue; SPSA and an applicable Bayesian optimizer for the later continuous-pulse comparison.
3. A history-aware model-free policy with the same observations.
4. Planning with the best learned Markov model.
5. Planning with each qualified learned memory model, using the same planner.
6. A policy trained or distilled from a learned model.
7. Known-dynamics planning as a diagnostic ceiling under the common action constraints.

First compare planners in O0. Add policies acting during an episode only in O2, with real instrument backaction in the simulator. Do not give model-free controllers dense state-fidelity rewards that the model-based methods would have to estimate, or vice versa. Training in simulation and querying the hidden reference are different resources and must be separately counted.

Measure achieved infidelity, catalogue regret, predicted-versus-achieved reward, 95th-percentile failure, total preparations to a fixed target, and latency. Evaluate fixed-query budgets and a reuse curve over 1, 5, and 20 distinct control goals. Count characterization, tuning, validation, adaptation, and reward estimation. Compare a recurrent policy without prediction to the same policy with a predictive auxiliary objective to isolate the value of world modeling from the value of memory alone.

Perform a critical stress test: let the optimizer search for controls that the model predicts are unusually good, then evaluate those controls in the independent reference. This tests whether lower average forecasting error actually leads to better decisions. Add uncertainty penalties or fallbacks as ablations with their own costs.

Deliverable: control-versus-query curves, a forecast-error versus control-regret plot, and the measured break-even number of reuse tasks. A Markov model may win this track without any quantum-memory finding.

## R2-S / R3-S: Streaming signals, events, and timely response

The first streaming experiment asks whether a learned predictive state improves response to meaningful changes relative to strong causal filters and simple event rules. Keep three subtasks separate: denoise/estimate the current state; detect or predict an event probability; and choose a control that improves the physical outcome. A detector with high classification accuracy is not automatically a useful controller.

Start with four stream settings:

| Setting | Signal and event | Main counterexample/control |
|---|---|---|
| S1 | Continuous weak readout of a qubit with a prescribed measurement instrument | Measurement noise produces apparent excursions without a physical fault |
| S2 | A hidden classical detuning switch or slow drift observed through a permitted probe | A fitted Bayesian switching filter may be sufficient |
| S3 | Photon-count or syndrome stream with a heralded transition, burst, or informative absence | Detector dead time or missing packets can imitate no-click evidence |
| S4 | Readout outliers, saturation, delays, and dropped samples with unchanged device dynamics | Responding to a sensor artifact can damage an otherwise correct state |

Generate streams using independently checked stochastic-master-equation or quantum-jump dynamics where appropriate. Preserve the relation between measurement innovation and state backaction. Keep a separate classical detector-noise model. Define event onset from the simulator's mechanism or a prespecified operational criterion; do not invent unique hidden quantum jump times where the monitoring protocol does not define them. A no-event Poisson interval carries likelihood information but never guarantees that an event is imminent.

Use the same observations, integration windows, and control access for: threshold/hysteresis; a causal change detector such as CUSUM; an appropriate Bayesian filter, including a fitted quantum filter; a small GRU or causal convolutional policy; and a learned predictive model plus policy. A Kalman filter is appropriate only for its supported approximate linear/Gaussian cases. A known-parameter quantum filter is a privileged reference; compare learned models with estimated-parameter filters too. Event-driven or spiking architectures are optional later ablations if compute/energy measurements justify them.

Distinguish causal filtering from offline smoothing. Test prediction at the future actuation time to determine whether anticipating latency helps. Compare a single fast model with a fast filter/policy plus slower model adaptation. First hold weights fixed during each evaluation stream and update only the recurrent belief state. Then evaluate parameter adaptation under drift with a separately specified update schedule, time cost, and held-out future segments.

Measure end-to-end event-to-effect delay and decompose acquisition/evidence accumulation, transport/queueing, computation, and actuation. Record overlap from pipelining. Report p50, p95, p99, maximum observed latency, throughput, queue growth, and deadline misses; a maximum observed value is not a proven worst-case bound. Replay benchmarks must run at the specified arrival rate with batch size one or the explicitly permitted acquisition window. Distinguish emulated physical delay from measured processor runtime. A laptop benchmark cannot establish FPGA or sub-microsecond hardware performance.

Sweep imposed total response delay relative to a prespecified useful intervention window, initially at fractions {0, 0.05, 0.1, 0.25, 0.5, 1}. The zero-delay case is a reference only. Obtain the window from the mechanism's control opportunity, not automatically from T1/T2. Add isolated burst, jitter, and dropout stresses before combinations. Use exact timestamps for rare-event matching and count repeated alarms for one event according to a frozen rule.

The first screening target is at least 20% lower achieved control error than the best causal baseline at matched measurement cost, false alarms, and response deadline, or at least 2× faster event detection at matched recall/false alarms with demonstrated control benefit. As a provisional engineering target, test whether p99 response delay is below 0.1 of the useful window and deadline-miss probability below 0.001. These are proposed operating points, not hardware claims or universal requirements; estimate sufficient tail exposure and confidence before calling them satisfied.

For scale, zero misses in 3,000 independent opportunities gives a one-sided 95% binomial upper bound of approximately 0.001. Consecutive loop iterations can share load and timing disturbances, so they cannot automatically be counted as independent trials. Include separate runs and stress blocks; report uncertainty appropriate to their dependence.

Report false alarms per physical time, event precision/recall, detection delay including missed-event accounting, event-rate dependence, calibration, useful lead time, harmful interventions on artifact-only streams, and recovery after a disturbance. Fix the operating false-alarm rate from a task-level action-cost budget using pilot data, then freeze it. Per-sample accuracy and a generic 1% classification error can hide unacceptable numbers of false actions in a long high-rate stream. Match ROC operating points before comparing speed.

Recorded data can validate causal detection or replay latency under its original actions. To claim an intervention benefit, run fresh interactive simulations or prospective hardware: changing actions changes later measurements, so following arbitrary new actions through an unchanged recorded future is invalid. Candidate public trajectory data require an availability/schema audit; current terminal RB records are not a streaming benchmark.

Deliverables: event-aligned response plots; control error versus total delay; detection delay versus false-alarm rate; latency tails under load; and an ablation showing whether learned dynamics adds value beyond a causal estimator or a recurrent policy. Relevant benchmarks include [451 ns electronic-feedback latency with a 48 ns neural contribution](https://www.nature.com/articles/s41467-023-42901-3) and a [6.19 microsecond per-inference charge-jump detector](https://arxiv.org/abs/2607.14293); their timing boundaries and observation protocols must be reproduced before comparisons.

## R3: Determine what observations are worth acquiring

Use known mechanisms and the public multitime tomography release as complementary tests. The [NMN-tomo repository](https://github.com/Christina-Giar/NMN-tomo) contains experimental data and reconstruction code for ten reported settings. Audit preparations, instruments, shot information, and reconstruction uncertainty before choosing supported tasks. Reproduce a published quantity before presenting an extension.

Construct pairs of mechanisms that are hard to distinguish under the declared RB observations. Require a positive-control demonstration that some permitted richer probe distinguishes them. Compare random probes, fixed spectroscopy-inspired schedules, an established adaptive uncertainty/information criterion, and our proposed decision-focused criterion. Finite-ensemble disagreement is a heuristic, not a bound over all compatible processes.

Evaluate both model-family discrimination and downstream action selection. A probe can be valuable even if it does not uniquely identify the mechanism, provided it reliably changes the choice toward a better control. Conversely, a strong quantum-memory witness is scientifically valuable even without a large survival-RMSE gain.

Start with short horizons of two to four intervention times for compatible-process calculations. Preserve physicality, causality, SPAM assumptions, and numerical feasibility tolerances. If attempting a quantum-memory claim, implement the actual witness or valid relaxation, verify its null set, and optimize over the relevant uncertainty set. Failure to reject a finite menu of models is not a universal impossibility result. [Restricted-control witnesses and bounds](https://quantum-journal.org/papers/q-2025-04-08-1695/)

For a final discrimination study, calibrate a test at false-positive rate 1% and target power 90% at prespecified separated alternatives. Use an independent calibration set and at least 1,000 independent null datasets per claimed null family in the initial error-rate audit, with binomial intervals; increase replication if the interval cannot support the claim. At roughly 10 errors in 1,000 trials, a point estimate of 1% does not establish that the true rate is at most 1%. Correct for tested families/endpoints where a joint claim is made. Sequential adaptive decisions require a prespecified stopping rule or appropriate sequential inference.

Deliverable: measurement-budget versus discrimination-power and control-regret curves, with a map of ambiguous regions. Later hardware proposals select a small set of informative timings and interventions from this evidence; the earlier five-by-five hardware grid is a candidate pool, not a mandatory acquisition.

## R4: Independently test constraints and adaptive capacity

The simple Bloch-ball projection example does not answer the original representability hypothesis. Use a tiny exact fermionic system, initially four spin-orbitals at fixed particle number, and compare unconstrained prediction, physical parameterization, penalty constraints, post-hoc projection, and a parent-state decoder. Extend to six/eight orbitals only after checking indexing, contraction identities, and the exact teacher.

Match observations, rollout horizons, and tuning opportunity. Evaluate normalized one-/two-body observables, contraction error, particle number, positivity violations, and energy behavior. Apply energy-conservation tests only when the specified dynamics conserve energy. For driven or dissipative dynamics, compare to the correct balance/reference evolution. P/Q/G conditions are relaxations; satisfying them is not a general proof of N-representability.

A constraint result requires better forecast or decision performance, not only a smaller violation. Compare a constraint applied during learning with the identical transformation applied only after prediction. Include projection time and bias. Use held-out quenches and at least twice the training rollout length.

For adaptive memory/bond dimension, compare fixed capacities, an adaptive rule, and a budget-matched fixed schedule on R1 or a small spin chain. Use spectra/residuals as candidate indicators and test whether they predict actual observable errors. Gauge transformations should preserve the relevant predictions; raw learned tensor norms or noncanonical spectra need not be physical quantities.

Deliverables: observable-error versus violation plots and error-versus-runtime curves. A useful adaptive rule should reduce cost while controlling actual errors, even if its internal memory has no unique microscopic interpretation.

## R5: Preserve a bounded spatial/size-transfer branch

Run one minimal HOTRG/Ising gauge and observable-preservation study and one small-chain dynamics feasibility check within the allocated allowance. For HOTRG, start with bond dimensions 4 and 8 and algebraically equivalent input tensors. Score free energy and an independent correlation observable. For dynamics, begin with exact chains small enough to verify, such as lengths 6 and 8, and test size transfer separately from time transfer.

Expand to canonicalized RG predictors or tensorial recurrent networks only if the teacher, observables, and gauge conventions are reliable. Compare to simple observable regression or a suitable non-tensor predictor. HOTRG is a spatial contraction method; success on equilibrium quantities is not evidence of a controlled temporal world model.

Deliverable: a short feasibility report and a concrete next comparison, or a cost-based deferral labeled untested. This branch can become a separate paper rather than being forced into the quantum-control argument.

## Quantitative goals and decision rules

These targets are proposed minimum worthwhile effects for allocating the next research round. They are not literature-wide standards, observed noise floors, or guarantees of publishability. Freeze each primary endpoint and applicable tolerance before opening its evaluation split.

| Track/claim | Screening target | Additional condition for a confirmed result |
|---|---|---|
| Prediction | ≥20% reduction in reducible MSE versus strongest applicable comparator; also report raw error | Fresh process instances or acquisition blocks; interval supports a practically useful effect |
| Accurate reduced-state rollout | Mean trace distance ≤0.01 at 2× training horizon; reference error ≤0.001 | Hold out parameters/controls and report tails; compare with reference methods at matched cost |
| Control quality | ≥20% reduction in achieved infidelity at equal total query budget | Independent policy evaluation, common constraints, and no material degradation in prespecified tail risk |
| Measurement efficiency | ≥2× fewer total preparations to the same control or discrimination target | Include fitting/calibration/verification costs; baseline must reach the target or report censored budgets |
| Adaptive computation | ≥2× less runtime at the same observable tolerance | Include adaptation and any amortized training costs; audit tolerance violations |
| Streaming response | ≥20% lower control error at matched false alarms/deadline, or ≥2× faster detection with control benefit | Causal information only; account for evidence accumulation and actuation, latency tails, missed events, and sensor artifacts |
| Constraint benefit | ≥20% reduction in a prespecified observable/rollout error at comparable cost | Gain survives a strong unconstrained model and post-hoc projection comparator |
| Quantum-memory evidence | A valid witness/bound under stated controls and uncertainties | Calibrated null error and prospective/statistically independent evidence; prediction gaps alone are insufficient |

Define reducible risk in simulation as \(E_m=L_m-L_{\rm oracle}\), with losses scored on the same distribution and the oracle using the true outcome probabilities conditional on the same available observations. Marginalize unobserved noise/environment states; an oracle that secretly knows their realization is a different privileged ceiling. Compare \(1-E_m/E_b\) only when \(E_b\) is materially positive. Finite-sample estimates can be negative; report that uncertainty rather than hiding it by clipping. Without a justified experimental noise floor, use paired raw risks and do not label them reducible. For analog readout, the oracle loss must use its observation model, not an invented binomial model.

For a terminal-probability target, 0.001 is a useful noiseless implementation tolerance. It does not establish acceptable gate error for fault tolerance. For state errors, trace distance 0.01 bounds each measurement-event probability error by 0.01; that gives this target an operational interpretation.

For an improvement \(\Delta\) with minimum worthwhile effect \(\Delta_*\), a confirmed GO requires its appropriate lower confidence bound to exceed \(\Delta_*\). An adequately powered NO-GO for that effect requires the upper bound to be below \(\Delta_*\). Overlapping bounds mean unresolved. Exploratory screening can justify another round without meeting the confirmation rule; record that decision separately.

Choose sample size from pilot variability, the chosen effect, clustering, and the desired power. Use independent mechanism draws for simulation generalization, independent trajectory draws for shot uncertainty, and acquisition/calibration blocks for hardware reproducibility. These are different levels. Eight or twelve blocks are possible starting budgets, not guarantees of a useful confidence interval. Near-zero reference errors require an absolute margin; unstable relative percentages do not define success.

Record a prespecified noninferiority margin for tail failure and inactive regimes based on the task. Do not universalize the earlier 0.005/0.010 RB RMSE margins to unrelated control tasks. Likewise, a fixed number of optimizer seeds, a 3σ deviation, or a fitted mixing weight is not a substitute for a valid statistical test.

## Execution stages and review artifacts

1. **Audit and reference checks.** Finish R0; test teacher solvers and observation contracts. Produce the initial coverage table and run manifest. No model-family rejection is allowed at this stage.
2. **Recovery and low-cost screen.** Run easy positive controls and the 18 development instances. Record all failures and the three-recipe development allowance. Pilot runtime before choosing larger jobs.
3. **Complete the directional matrix.** Freeze settings; run the 12 first-screen instances, common control tasks, observation comparisons, streaming response/delay tests, and the independent R4/R5 pilots. This is the first review point for reallocating effort.
4. **Expand informative directions.** Run data/compute curves and specific failure-boundary sweeps. Preserve strong baselines and an explicit misspecification challenge. A new architecture needs a written mechanism hypothesis and an ablation against its nearest simpler alternative.
5. **Fresh confirmation.** Freeze methods and endpoints; obtain fresh synthetic instances or a separately supported experimental evaluation. Determine replication by power analysis. If hardware access is unavailable, complete the simulation/public-data result and label the missing prospective hardware test.
6. **Publication artifacts.** Produce reproducible figures, run tables, claim-to-evidence links, limitations, and a manuscript/blog outline. No external publication or hardware job is part of merely drafting this plan.

For every run retain the data and code identifiers, full configuration, partitions, all initialization results, selected-model rule, predictions, targets, metrics, resource use, and a failure/status field. Figures should be generated from those records, with vector or high-resolution exports and scripts that rebuild them. Keep a chronological hypothesis ledger; logging a new hypothesis does not retroactively preregister old experiments.

Candidate articles include a baseline audit of apparent memory, an observation-versus-inference study, a prediction-versus-control study, and an independent constraints/compression study. A paper-level contribution additionally needs matched literature baselines, supported uncertainty, and a contribution beyond renaming existing methods. A blog can explain an exploratory result provided the evidence level and failed alternatives remain visible.

## The plan explained for a ten-year-old

Imagine trying to guide a tiny machine with buttons. You cannot see all of its parts, and sometimes pressing a button also changes a hidden part. Its display gives you only a noisy clue.

A world model is a practice version of the machine: “If I press these buttons, what is likely to happen?” A policy is the rule that chooses the next button. We might use the practice machine to test plans, teach the rule, remember clues, or choose a useful question to ask the real machine.

We already found that a small practice machine predicts one set of records quite well. It did not do as well under another setting. We also spent more time improving some practice machines than others. That means we should organize a fair competition before choosing our favorite.

First, we build small pretend machines whose hidden parts we know. Some forget immediately; some remember; some have a sticky button; some leak into an extra state. Each learning method gets a fair chance, and we check that it can learn an easy example before judging it on a hard one.

Next, we change what clues it gets. Does it see only the final display, get several separate measurements, or receive clues while it acts? In the quantum experiments, looking can change the machine, so that has to count as part of the experiment too.

Then we ask the most useful question: does the practice machine actually help choose better buttons? We compare it with simple rules and with a learner that practices directly. We count the cost of building and checking the practice machine as well as the trials it saves.

We also test clues that arrive continuously. A sudden flash could mean trouble or just a faulty sensor, and sometimes a missing flash is the useful clue. The controller must tell the difference and act before the chance to help disappears. We score how often it catches real problems, how often it presses a button for no reason, and how long the entire response takes. A clever answer that arrives too late does not win.

We also test whether obeying physics rules makes predictions useful for longer, and whether a smaller notebook can remember enough. These tests get their own chances; they do not lose automatically because another experiment went badly.

A green light means the method helps enough to justify more work. A red light means a fair test showed it does not help enough in that situation. A question mark means we need better training, better clues, or more evidence. Our final map should explain which machine wins where, how much it helps, and why.
