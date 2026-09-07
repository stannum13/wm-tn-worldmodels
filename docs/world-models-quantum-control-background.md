# Why explore world models for quantum control?

Research background note, 7 September 2026. This is a rationale and evidence review, not a report of new experiments. The associated [redistributed experiment plan](redistributed-experiment-plan.md) specifies the next studies. Published demonstrations, our exploratory results, and proposed contributions are distinguished throughout.

## The research proposition

A useful quantum controller must choose an action despite incomplete knowledge of the device, its environment, and its current condition. A learned world model could help by predicting the consequences of candidate actions from the information that the controller actually has. Its value would be demonstrated by better control, fewer measurements, faster decisions, or more reliable adaptation.

This proposition does not require a large neural network, a uniquely reconstructed microscopic environment, or computation on a quantum computer. A calibrated Hamiltonian, a dissipative channel, a classical state estimator, or a compressed process tensor could serve as the model. The research question is which representation is sufficient for a specified decision, and whether learning and using it costs less than the alternatives.

There is already substantial precedent. Learned models have supported quantum control; model-based reinforcement learning has reduced interaction counts in numerical studies; and history-dependent feedback has improved simulated error correction. The unresolved opportunity is to establish when these benefits survive restricted observations, model error, changing device conditions, and a fair accounting of all measurements and computation.

Our working hypothesis is therefore:

> A compact predictive model of experimentally accessible histories can improve quantum decisions when its prediction errors are small in the action regions that matter, its uncertainty is informative, and its acquisition and maintenance costs are amortized across useful decisions.

Each condition is testable. None follows simply from an attractive latent representation or a low average forecasting error.

## What “world model,” “policy,” and “memory” mean here

In machine learning, a world model is a learned representation of an environment's dynamics that can support imagined action sequences and controller training. Ha and Schmidhuber's *World Models* is an early explicit example of separating a compressed dynamics model from a controller. Its demonstrations are not quantum-control evidence; it supplies a useful architectural vocabulary. [Ha and Schmidhuber, 2018](https://arxiv.org/abs/1803.10122)

Let the available history be

\[
h_t=(o_0,a_0,o_1,\ldots,a_{t-1},o_t),
\]

where an observation may be a terminal readout, an ancilla syndrome, measurement statistics from repeated preparations, or a calibration record. An action may be a pulse, idle interval, measurement, reset, or gate choice. Timing and measurement settings belong in the history when they affect the dynamics.

A predictive representation and model might have the form

\[
z_t=f_\theta(h_t),\qquad
\widehat p_\theta(o_{t+1:T}\mid h_t,\operatorname{do}(a_{t:T-1})).
\]

The intervention notation emphasizes that the query asks what would happen if those actions were performed. Generalization to such queries requires suitable experimental coverage and assumptions; ordinary supervised prediction does not automatically identify them.

A policy is a different object:

\[
\pi_\phi(a_t\mid z_t,g),
\]

where \(g\) specifies a goal. The model predicts; the policy chooses. A single network can contain both components, but their scientific roles remain distinct.

| Role of the model | How it helps choose actions | What we must measure |
|---|---|---|
| Pulse or circuit simulator | Search candidate controls offline, then execute a chosen sequence | Achieved fidelity, measurement cost, robustness |
| Model predictive controller | Replan after receiving an allowed observation | Closed-loop reward and decision latency |
| Training environment | Train a policy using model-generated trajectories | Real interactions saved and performance after transfer |
| Policy distillation teacher | Convert a slower planner into a fast action rule | Speed, imitation error, and retained control performance |
| History/state estimator | Summarize observations for a controller or decoder | Incremental value over simpler filtering and recurrent policies |
| Experiment selector | Choose probes that reduce uncertainty relevant to a decision | Measurements needed for a fixed inference or control target |
| Uncertainty model | Detect unsupported plans and select a fallback or recalibration | Coverage, harmful-action rate, and cost of abstention |

Three meanings of memory must not be conflated. **Physical memory** is information retained in unobserved degrees of freedom that can later affect the system. **Estimator memory** records what previous observations tell us. **Policy memory** makes actions depend on history. A controller may benefit from estimator or policy memory even when the underlying physical dynamics are Markovian: noisy partial observations can make a single latest measurement insufficient.

Conversely, an accurate memoryless model is still a potentially useful world model. Discovering quantum environmental memory is a separate, stronger objective.

## Streaming observations, sudden events, and response time

Many useful quantum-control problems involve a live stream rather than a completed experiment. Inputs can include analog I/Q readout, photon counts, ancilla syndromes, or asynchronous calibration events. An observable's estimated expectation is not the same object as a raw single-run signal: shot noise, detector bandwidth, integration windows, and measurement backaction determine what can be inferred and when.

A spike may indicate a physical jump, leakage, a control disturbance, or a detector artifact. A missing expected event can also be informative. In an engineered three-level superconducting system, Minev and colleagues used the absence of the expected auxiliary bright-state signal to track and reverse a quantum jump during its flight. This is an experimental precedent for timely intervention using conditional information; it does not mean that arbitrary random quantum outcomes can be predicted. [Minev et al., Nature 2019](https://arxiv.org/abs/1803.00545)

Learned filters are relevant here. Flurin and colleagues reconstruct superconducting-qubit trajectories from continuous measurement records with recurrent networks. For a controller, the available estimate must use only measurements already received. An estimate improved using later samples is a smoother and must be reported separately from a causal filter. [Flurin et al., PRX 2020](https://arxiv.org/abs/1811.12420)

There are concrete hardware timing anchors. Reuer and colleagues demonstrate a neural policy for superconducting-qubit initialization with a reported electronic feedback latency of 451 ns; pipelined neural computation contributes 48 ns of that latency. Acquisition and integration windows remain part of the overall experimental timing. A July 2026 charge-jump detection preprint reports 6.19 microseconds per FPGA inference and detection efficiency 0.843 ± 0.022, compared with 0.866 ± 0.020 for its offline comparator at matched false-positive rate in the stated jump-amplitude range. That detector was trained on synthetic scans based on measured templates; per-inference latency is not the complete time from a physical jump to correction. [Reuer et al., Nature Communications 2023](https://www.nature.com/articles/s41467-023-42901-3), [Gaytan-Villarreal et al., preprint 2026](https://arxiv.org/abs/2607.14293)

For streaming control, the model has four possible jobs: maintain a belief about the current state, estimate the likelihood of a consequential event, forecast the state at the time an action will actually arrive, and update slower parameters as evidence of drift accumulates. These jobs can operate at different rates. A small causal filter and policy could handle the fast loop while a larger model periodically updates calibration or distills a new policy. This is a candidate organization to test, not a presumption that two models are necessary.

The operative quantity is event-to-effect delay. It includes the time needed to accumulate evidence, communication and queueing, computation, and actuator response. Pipelining can overlap components, so instrument timestamps rather than summing incompatible reported latencies. A policy that is accurate after the useful intervention window is ineffective, and a smoother that uses future data cannot establish responsiveness.

Our proposed streaming objective is therefore useful response at a specified false-alarm rate, measurement strength, and physical deadline. Report event recall, false alarms per unit time, detection delay, tail latency, missed deadlines, and achieved control improvement. Compare against threshold/hysteresis rules, classical change detection, appropriate Bayesian/quantum filters, and small recurrent policies. A predictive model earns its cost only if it improves this trade-off. Sparse or event-driven networks can enter as computational alternatives, with energy or speed advantages measured on the actual execution target.

## Why the quantum setting makes the proposal plausible—and difficult

A reduced density matrix can omit information relevant to future evolution. Two joint system–environment states may have the same system marginal and different future responses. The process-tensor framework describes the statistics generated by sequences of interventions; operational tests of Markovianity ask whether relevant history dependence survives appropriate measure-and-reprepare interventions. [Pollock et al., 2018](https://arxiv.org/abs/1801.09811)

This suggests learning a predictive state containing just the missing information required by the task. It does not imply that the learned coordinates uniquely describe a real bath. Nor does it imply that every process has a compact representation. Long-lived correlations, leakage, or complex environments can make the necessary representation large.

Measurements introduce an additional constraint. The controller cannot repeatedly read an unknown quantum state without disturbance. An experiment with terminal readout normally supports control updates between repeated runs. A feedback experiment with ancilla measurements supports updates during a run, but those measurements and their backaction must be part of the dynamics. Ensemble tomography requires multiple preparations and their cost must be counted.

For a physical instrument with outcome \(o\), a conditional state update has the form

\[
\sigma'_{SE}=\frac{\mathcal J^{a}_{o}(\sigma_{SE})}
{\operatorname{Tr}\mathcal J^{a}_{o}(\sigma_{SE})},
\quad p(o\mid a)=\operatorname{Tr}\mathcal J^{a}_{o}(\sigma_{SE}).
\]

Here the instrument includes the specified evolution and measurement. A simulator may know \(\sigma_{SE}\); a deployed controller generally does not. Giving it that state would answer a different, easier question.

Restricted controls also limit identification. Unitary sequences followed by terminal measurements can leave multiple multitime processes compatible with the observations. Nevertheless, valid witnesses and upper/lower bounds on temporal properties can sometimes be obtained under those restrictions. Our task is to determine which predictions or properties are supported, rather than assume complete identification. [White et al., 2025](https://quantum-journal.org/papers/q-2025-04-08-1695/)

An RB dataset is consequently one observation regime within the program. It cannot by itself settle the value of learned memory under tomography, syndrome histories, or different controls.

## Five mechanisms by which a model might earn its cost

**Replacing expensive trials with useful calculations.** Pulse optimization may require evaluating many alternatives. Once fitted, a differentiable model can evaluate candidates and gradients without a new device experiment for each candidate. Khalid and colleagues report approximately an order-of-magnitude reduction in interactions relative to standard model-free RL for selected one- and two-qubit gate tasks in numerical closed/open-system studies. They use a Hamiltonian ansatz with known control dependence, so this result motivates a comparator rather than establishes arbitrary latent-model superiority. [Khalid et al., PRR 2023](https://arxiv.org/abs/2304.09718)

**Remembering information that changes a decision.** A sequence of noisy observations can reveal detuning or the probability of an error better than the last observation alone. In GKP error correction, Puviani and colleagues train recurrent feedback using a differentiable simulator and the outcome history. For a reported logical-state example, the simulated lifetime increases from about 700 to 1,500 correction cycles. This supports history-aware control; it is not a demonstration that a learned environmental world model is necessary or that the gain was achieved on hardware. [Puviani et al., PRL 2025](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.134.020601)

**Reusing dynamics across goals.** The same estimated response model may support several gates, target states, or noise-suppression tasks. This can amortize characterization, whereas independently optimizing every task repeatedly pays exploration costs. Driven neural quantum propagators already investigate reuse across initial states and driving conditions, so our comparison must include an applicable propagator or system-identification method. [Zhang, Benavides-Riveros and Chen, PRR 2025](https://arxiv.org/abs/2410.16091)

**Correcting imperfect plans using new observations.** Receding-horizon planning can reduce the consequences of model mismatch by updating its state estimate and replanning. Goldschmidt and colleagues demonstrate this numerically for a qubit, a weakly anharmonic system, and crosstalk. Their result motivates testing feedback under an explicit observation protocol; it does not make quantum state feedback free. [Goldschmidt et al., Quantum 2022](https://quantum-journal.org/papers/q-2022-10-13-837/)

**Learning which experiment matters next.** If plausible models recommend different controls, a targeted probe can resolve that disagreement. A useful probe should reduce expected decision loss, not merely separate arbitrary latent parameters. Adaptive quantum process tomography has experimental precedent, and measurement-time optimization has been studied for non-Markovian dephasing. Our proposed extension is a comparison under competing mechanisms, calibration uncertainty, and a fixed total budget. [Pogorelov et al., PRA 2017](https://arxiv.org/abs/1611.01064), [Varona et al., 2025](https://www.nature.com/articles/s41534-025-01044-7)

A simple physical example keeps these motivations grounded. For an ideal qubit with a static unknown detuning, \(H=\delta Z/2\), free evolution is \(U(\tau)=\exp(-i\delta\tau Z/2)\). Ideal instantaneous inversion pulses satisfy \(XU(\tau)XU(\tau)=I\). Echo therefore cancels this error without learning a sophisticated model. It must be a baseline. A model might become useful when detuning varies, pulses have duration and distortion, or several noise sources compete. Whether that additional knowledge improves on standard pulse sequences is the experiment.

## What the closest literature has already achieved

These are task-specific anchors, not a shared leaderboard. Different fidelity definitions, hardware, information access, training budgets, and cost denominators prevent direct numerical ranking.

| Study | Evidence and quantitative anchor | Implication for this project |
|---|---|---|
| [Baum et al., PRX Quantum 2021](https://arxiv.org/abs/2105.01079) | Hardware experiment: single-qubit gates up to 3× faster than default DRAG without added leakage; cross-resonance gates also show improved performance in tests up to 25 days after optimization | Direct RL and direct experimental optimization are serious baselines; learning dynamics is optional |
| [Youssry et al., npj QI 2024](https://www.nature.com/articles/s41534-023-00795-5) | Experimental graybox identification and control of a voltage-controlled photonic circuit | Physics plus flexible inference is established; its benefit must be tested in our observation regime |
| [White et al., PRX 2025](https://arxiv.org/html/2312.08454v2) | Self-consistent tensor-network characterization on IBM devices. One application uses 3,000 circuits and approximately 3 million shots per implementation, with a reported reconstruction error around \(10^{-4}\) | Compare against self-consistent characterization, not only ideal-control OQE. The paper distinguishes model-predicted control improvements from live feedback deployment |
| [Zhang et al., Communications Physics 2025](https://www.nature.com/articles/s42005-025-01944-2) | Experimental RB-based open-quantum-evolution learning and forecasting; public controls, measurements, and reconstruction artifacts | Direct forecasting comparator, with protocol matching required; the paper already discusses dissipative embeddings |
| [Zhu et al., npj QI 2026](https://www.nature.com/articles/s41534-026-01269-0) | Numerical representation-guided RL with limited measurement settings, including a 50-qubit ground-state task | A learned observation representation can assist a policy without a predictive world model. The protocol uses multiple copies to obtain statistics at control steps |
| [Zhong et al., June 2026 preprint](https://arxiv.org/abs/2606.27907) | Joint latent dynamics/control learning on numerical two-level and spin-chain tasks; reports roughly 1,000× lower optimization cost | Joint latent control is already an explicit proposal. Reproduce cost definitions, include training, and test unknown rather than supplied environmental parameters before making a speed claim |

The close literature rules out claiming novelty merely from combining neural networks, memory, physical structure, and control. A contribution would need a demonstrated improvement in a specified setting: fewer measurements, more robust decisions, useful uncertainty, broader transfer, or a defensible boundary on what can be learned.

## Why good prediction may still produce bad control

A planner searches for high predicted reward. That search can discover model errors as effectively as it discovers good controls. Consequently, a model can predict random test sequences accurately while recommending an unusually bad sequence outside its reliable region.

The following elementary bound makes the issue precise. Let \(J(a)\in[0,1]\) be true terminal success over a declared candidate set \(\mathcal A\), and let \(\widehat J(a)\) be the model estimate. If

\[
\sup_{a\in\mathcal A}|J(a)-\widehat J(a)|\leq\epsilon,
\]

then a planner whose model reward is within \(\delta\) of the model optimum has true regret at most \(2\epsilon+\delta\). This follows by inserting the model reward between the true rewards of the optimal and chosen controls and bounding the two prediction errors. The statement concerns the declared candidate set, not an unknown global optimum over all physical controls.

The useful object is therefore error on decisions the planner might select. Average RMSE over an unrelated distribution does not provide this bound. Likewise, a model need not reconstruct every microscopic property if it ranks feasible controls correctly. Our experiments should report candidate ranking, selected-control regret, and the gap between predicted and realized reward alongside forecasting error.

For physical state predictions, trace distance \(D(\rho,\hat\rho)=\tfrac12\|\rho-\hat\rho\|_1\) bounds the probability error for any event measured by an effect \(0\leq M\leq I\). This gives a tolerance such as \(D\leq0.01\) an operational meaning. Achieving that bound on average over states still does not establish it uniformly over optimized controls.

Uncertainty penalties and ensembles are candidate defenses, not automatic certificates. An ensemble of similarly misspecified models can agree and be wrong. Test uncertainty on omitted mechanisms, shifted controls, and planner-selected actions, and measure the price of falling back to a conservative controller.

## The economic question: when is learning worth doing?

For a claim of fewer measurements at matched performance across \(K\) control tasks, the resource comparison is

\[
N_{\rm characterize}+N_{\rm calibrate}+N_{\rm validate}
+\sum_{j=1}^{K}N_{\rm adapt,j}
<\sum_{j=1}^{K}N_{\rm direct,j}.
\]

All terms count physical preparations or shots under a specified protocol. Report distinct circuits, shots, device time, and classical computation separately. A simulated environment interaction is not automatically one hardware shot; estimating a reward or state can require many preparations.

The inequality is a proposed accounting rule. Its terms must be measured. Reuse across tasks may favor a model, while rapid drift may erase that advantage by forcing frequent refits. Similarly, policy distillation can make inference fast after substantial offline training. Both its one-time cost and its reuse count matter.

The most plausible early setting is a small accessible subsystem, a repeatable family of tasks, expensive experimental queries, and a reasonably stable low-dimensional response. A known, cheap, accurate simulator or a simple robust pulse may already be sufficient. Large unobserved many-body environments, strongly changing calibration, or weakly informative observations can make learning uneconomical.

## Why investigate these particular primitives?

Physical composition, explicit memory, reduced states, and tensor networks offer different possible benefits. They should receive separate ablations. A generic recurrent predictor remains a legitimate alternative in every task where it can consume the same observations.

A channel model composes controlled transformations using the structure of quantum evolution. This can reduce the amount of data needed to learn how errors combine along a new sequence. However, enforcing the wrong time dependence, idealizing distorted pulses, or omitting leakage can make a physically valid model systematically inaccurate. Physical validity restricts what a model may predict; it does not establish that the model describes the device.

Auxiliary states and temporal tensor networks offer ways to retain information that a reduced state discards. They may be efficient when the relevant temporal correlations can be compressed. A useful test asks how much memory is required to preserve chosen observables or control decisions at a specified error. Environmental dimension, channel Kraus rank, and tensor bond dimension are different resources; a finite unitary reservoir can absorb ordinary dissipation as well as persistent memory. Reusable process-tensor simulation is already available, so a learned replacement must show an error–cost benefit over that reference. [OQuPy](https://arxiv.org/abs/2406.16650)

Reduced density matrices can expose several observables through one representation. Constraints on positivity, consistency, or particle number may prevent rollout errors from driving the predictor into impossible states. They can also introduce bias or expensive projections. Our hypothesis is that constraints improve future observables and decisions; producing nicer matrices is an insufficient endpoint. A predictive latent used directly by a policy need not itself be a density matrix.

Spatial RG representations have a more distant motivation: perhaps a compressed description preserves information relevant across system sizes. That is a separate hypothesis from temporal memory. Equilibrium contraction accuracy alone supplies no evidence that such a representation can predict controlled dynamics, which is why the plan assigns it a small independent feasibility study.

## What our experiments establish so far

These are exploratory forecasting results from a particular public release. They are not quantum-control or policy results.

| Observation | Achieved number | Supported conclusion |
|---|---:|---|
| `idle100`, biases 0.40–0.54, mean per-bias forecast RMSE | RB 0.15435; damped D1 0.08189; general Markov CPTP 0.08294 | A compact memoryless model captures substantial sequence structure |
| Same active bias group at `idle180` | RB 0.15454; damped D1 0.15457; CPTP 0.15487 | The frozen method did not reproduce the active gain after retraining in this condition |
| Pure-unitary OQE at `idle180`, bias 0.50 | D6 0.22431 versus RB 0.17221 | This implementation and training recipe remain uncompetitive there |
| Sequence correspondence, released D1 hybrid at `idle100`, bias 0.50 | Within-length correlation approximately 0.906 | The improvement includes sequence-specific information, not only the mean decay |

The 100 ns versus 180 ns comparison was **method transfer with retraining**, not zero-shot transfer of parameters. Both conditions have now been explored and cannot supply new confirmation for further choices. The matched memoryless comparison used 4,680 training rows, 3,120 validation rows, and 4,000 length-41–60 forecast rows per tested condition; experimental independence must be assessed at acquisition/calibration level, not inferred from row count. [Local benchmark findings](memoryless-benchmark-findings.md), [scientific review](scientific-review.md)

Our search was uneven. The two memoryless families received 100 fits across conditions. The quasi-static classical family received ten fits at one difficult condition across two objectives. Some dissipative-memory models collapsed toward the mean without passing a mechanism-recovery benchmark. Generic models were compared using a different horizon protocol. These facts justify reopening those comparisons, not assuming that more work would necessarily make them win.

No new policy has yet improved a quantum-control task in this repository. The physically constrained Bloch and scalar-memory controls are implementation demonstrations. They do not establish the benefits of fermionic representability, adaptive tensor bonds, or counterfactual quantum control. [Earlier process critique](process-critique.md)

## The research directions that follow

The scientific opportunity is to measure when predictive representations become useful for decisions. Five questions organize the work:

1. **Sufficiency:** Does history improve action ranking after adequate Markovian, classical, control-error, and calibration alternatives are included?
2. **Control value:** At equal observation and action access, do learned-model planning or model-trained policies improve on direct optimization and recurrent policies?
3. **Information value:** Can selected probes reduce the measurements needed for reliable decisions or valid memory tests?
4. **Constraint value:** Do physical constraints improve future observables and controls, beyond making outputs formally valid?
5. **Compression value:** Can a smaller or adaptive representation preserve these benefits at lower total cost?

Controlled simulations can distinguish failures of information, optimization, representation, and decision-making. Existing experimental data can test applicability under actual measurement limitations. Prospective hardware tests can establish reproducibility and achieved control improvements. These forms of evidence have different jobs.

OQuPy supplies an existing route to process-tensor reference dynamics and control calculations. QD3SET-1 includes 1,000 HEOM spin-boson trajectories through \(t\Delta=20\), useful for dynamics and bath-parameter transfer. Additional interventions or multitime observables must be generated if the released trajectories do not contain them. [OQuPy](https://arxiv.org/abs/2406.16650), [QD3SET-1](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2023.1223973/full)

A useful paper could establish a favorable region of measurement cost, memory strength, and control transfer. It could also establish a carefully delimited negative result: a simple estimator matches a complex model, an apparently good predictor recommends poor controls, or accessible measurements leave decision-relevant ambiguity. A failed fit alone cannot establish any of those conclusions.

The associated plan treats these as independent routes to evidence. Its central comparison is the smallest model or policy that achieves a specified control objective with defensible resource and uncertainty accounting.
