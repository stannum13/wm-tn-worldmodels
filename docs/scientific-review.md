# Independent scientific review and revised experiments

Date: 2026-09-06. An independent AI reviewer inspected implementation, evaluation,
upstream acquisition code, and primary literature. This document synthesizes its
critique with explicitly identified diagnostics run by the parent agent. Neither
agent has human scientific credentials; novelty and device mechanisms are not
established by this review.

## What survives the review

The released OQE predictions contain useful sequence-specific information. A small
calibrated D1 predictor can outperform larger uncalibrated models on the already
explored survival-probability forecasts. D1 has no environmental memory. These
observations support predictive compression, not a claim that environmental memory
has been identified or that the recovered process is correct for arbitrary controls.

The sharper research question is:

> After accounting for dissipation, control calibration, and measurement, what
> persistent information is necessary to predict a controlled quantum process—and
> which additional interventions can distinguish competing explanations?

"Necessary" must first be stated relative to a declared model class. A device-level
quantum-memory claim additionally requires a valid witness or bound, not merely a
lower loss from one fitted architecture.

## Ranked critique

### 1. Missing dissipation is plausible, but is not a new mechanism

The current [OQE implementation](../src/ptwm/oqe.py) evolves a pure system-environment
state with a repeated unitary. D1 cannot irreversibly contract the Bloch ball. A larger
unitary environment can absorb effects that an ordinary dissipative channel represents
more compactly; its fitted dimension is therefore not automatically physical memory.
The target paper explicitly discusses this inefficiency and proposes dissipative
system-environment channels as an alternative. Adding damping is a necessary comparison,
but this idea alone is not a novel extension of the paper.
[Zhang et al., OQE section](https://www.nature.com/articles/s42005-025-01944-2)

### 2. The memoryless comparator must have adequate channel capacity

A qubit interacting with a fresh pure D2 ancilla implements a channel of Kraus rank
at most two. General qubit CPTP channels can have rank four and have 12 real degrees
of freedom in their interior. Comparing persistent D2 solely with reset D2 can confuse
inadequate one-step channel capacity with a benefit from retaining memory.

Include a general qubit CPTP channel, with the same preparation/readout treatment.
Start with time-homogeneous noise interleaved with known controls. If gate-duration
or gate-dependent errors remain plausible, add a constrained control-dependent
Markov comparator before interpreting a memory win. A homogeneous-model rejection
does not reject every memoryless explanation. General-channel capacity is a
structural comparator, separate from a matched-parameter/computation comparison.

### 3. Clifford algebra does not fix laboratory preparation and measurement axes

[clifford.py](../src/ptwm/clifford.py), lines 111–134, selects the first valid group
isomorphism. [oqe.py](../src/ptwm/oqe.py), lines 48–57, fixes the initial state and
measurement axis to `|0>`. Conjugating gates is a harmless coordinate change only
when noise, state, and POVM transform together. Closure checks establish a consistent
abstract representation, not a unique laboratory-axis assignment.

Resolve the joint control/SPAM gauge before extracting physical detunings or coupling
directions. A valid conjugation test must transform every object and leave predictions
unchanged. A separate sensitivity test should vary unresolved relative alignments.

The stored Hermitian generator also retains one unobservable identity/global-phase
direction: `(2D)^2` is a stored parameter count, not a minimal identifiable count.
D1 has at most three observable unitary-generator directions.

### 4. Previous cross-architecture claims used different protocols

[run_architecture_sweep.py](../scripts/run_architecture_sweep.py), lines 103–105,
trains on lengths at most 20 and tests longer lengths. In contrast,
[train_reconstructed_oqe.py](../scripts/train_reconstructed_oqe.py), lines 104–108,
uses random disjoint sequences across lengths 2–40 for training and validation,
then forecasts lengths 41–60. Therefore these runs cannot establish that Clifford
geometry caused the model-to-model performance difference. Repeat architecture
ablations on identical records, horizons, calibration, and tuning budgets.

The OQE training script also uses the same validation set for early stopping and
calibration. That is a development choice, not a fresh validation of the calibrator.
The 0.5 gate and dimension choices were explored on the existing bias sweep. Reserve
new conditions for confirmation; do not present repeated old-test improvements as
independent discoveries.

The coefficient threshold is not a statistical confidence test. When clipping is
inactive, doubling a correction halves its fitted coefficient without changing the
calibrated prediction. A weight of 0.7 can become 0.35, changing activation under
the current rule even though the useful prediction is identical. Assess activation
using out-of-fold risk improvement and its uncertainty instead of correction scale.

### 5. Measurement and uncertainty models need acquisition-aware treatment

Upstream acquisition code constructs `p0` by projecting analog complex readout onto
calibration points. It is not explicitly a binomial fraction even though acquisition
uses repeated shots. The same script loops over sequence lengths within acquisition
runs, suggesting dependence from shared calibration and drift. Do not substitute a
binomial likelihood without reconstructing the measurement pipeline.
[Pinned acquisition code](https://github.com/guochu/pt_recovery/blob/47d67598a304bb72c315bf05ddfadcda5f4be290/experiment_data/RB_data_20230104/len60/idle180/Q67_flux_sq_rb_script.py#L41)

The row bootstrap in [compare_released_oqe.py](../scripts/compare_released_oqe.py),
lines 37–47, does not represent uncertainty across calibration sessions. Recover
acquisition grouping and resample blocks if possible. Otherwise state that the
interval is conditional on rows and cannot establish across-session reproducibility.

Pulse and idle intervals must be separated in a shared-time model. Upstream timing
includes finite gate duration as well as idle time; changing `idle100` to `idle180`
is not equivalent to multiplying every learned step by 1.8.
[Pinned timing implementation](https://github.com/guochu/pt_recovery/blob/47d67598a304bb72c315bf05ddfadcda5f4be290/experiment_data/RB_data_20230104/len60/idle180/flux_sq_rb.py#L34)

## New diagnostics performed during this review

These are parent-agent calculations on released forecasts, not independent reviewer
training runs. No OQE generator was retrained. Source commit:
`47d67598a304bb72c315bf05ddfadcda5f4be290`.

### Sequence signal survives removal of length effects

Keep the original validation-fitted hybrid weights. Decompose predictions into their
mean within each length plus a centered sequence-specific component. For released D1
at bias 0.5, forecasting lengths 41–60:

| Diagnostic | RMSE or stated quantity |
|---|---:|
| RB | 0.183408 |
| Existing D1 hybrid | 0.087995 |
| Hybrid predictions averaged within length | 0.184388 |
| Square root of expected MSE after within-length permutation | 0.272293 |
| Correlation after centering targets and predictions within length | 0.905920 |

The permutation statistic is computed analytically, not from a selected shuffle.
It is `sqrt(E[MSE])`, not `E[RMSE]`. Prediction averaging uses no target values and
is an evaluation decomposition, not a separately trained deployable model. At all
four active biases, D1's mean correction worsens RB MSE; the within-length sequence
correction supplies the gain. This establishes sequence matching conditional on
the release's prediction/label alignment, not environmental memory.

### Fixed-generator D1 with damping and output calibration

Isotropic depolarization commutes with every unitary rotation. Given a released D1
prediction `q(s)` and `m = len(s) + 1` noise steps, the exact damped-D1 prediction is

`p(s) = 1/2 + eta^m * (q(s) - 1/2)`.

Apply bounded affine output calibration `a*p + b*(1-p)` with `a,b,eta` in `[0,1]`.
The affine map has the mathematical form of classical readout confusion, but fitted
parameters need not correspond to the actual analog-readout pipeline or device
readout errors. Fit these parameters on released validation lengths 21–40 only.
Three initial retention values are compared using validation loss only.

| Bias | Existing D1 hybrid | D1 + affine calibration | D1 + damping + affine calibration |
|---|---:|---:|---:|
| 0.10 | 0.05499 | 0.07876 | 0.06488 |
| 0.40 | 0.09136 | 0.09574 | 0.08674 |
| 0.50 | 0.08800 | 0.08714 | 0.07739 |
| 0.52 | 0.09093 | 0.08940 | 0.07766 |
| 0.54 | 0.09859 | 0.09672 | 0.08184 |
| 0.60 | 0.06950 | 0.07017 | 0.07012 |

The existing hybrid falls back to RB at 0.1 and 0.6. New damping/calibration improves
all four explored active conditions and is worse than RB at both inactive controls.
Boundary-hitting calibration estimates caution against interpreting the fitted
parameters as identifiable physical quantities. These results motivate joint
training and independent confirmation; they do not establish universal superiority.

Reproduce with:

```bash
python scripts/audit_released_d1.py data/external/pt_recovery \
  --output results/review_diagnostics.json
```

Results: [review_diagnostics.json](../results/review_diagnostics.json).
The permutation formula was checked against exhaustive permutations on a toy case;
the damping formula was checked against explicit Bloch-vector evolution at three
sequence lengths.

## Revised minimal experiment sequence

| Order | Experiment | Competing predictions and decision |
|---|---|---|
| 1 | Gauge-consistent, jointly trained damped D1 and general qubit CPTP baselines | If these match the best forecasts, ordinary coherent/dissipative noise explains the predictive gain within this task. Do not claim memory necessity. |
| 2 | Compare D2 memory retained, completely dephased, and reset between steps | If dephasing retains accuracy, inter-step environmental coherence is unnecessary within this model class. If reset retains accuracy, persistent memory is unnecessary. Compare with the full CPTP baseline in either case. |
| 3 | Add a classical correlated-noise model using control geometry | If it matches persistent D2, that fit does not establish uniquely quantum memory. Test multiple classical capacities and physical correlation times, not one underpowered competitor. |
| 4 | Freeze model selection and confirm on idle180 and held-out biases | Distinguish retraining transfer from zero-shot physical-time transfer. Report active-condition improvement and inactive-condition harm separately. |
| 5 | Use short-horizon compatible-process bounds and discriminating control probes | Seek controls where physically admissible competing explanations disagree beyond uncertainty. A valid memory witness, if obtainable, is stronger than architecture ranking. |

For experiment 2, retain/reset/dephase are interventions on the simulated memory,
not interventions that this dataset performed on the physical environment. Compare
both fixed-parameter ablations and independently retrained variants. Dephasing is
basis-dependent; optimize or systematically vary its allowed basis. A dephased D2
model is one particular two-state classical-memory family, not every possible
classical explanation. An ablation gap alone is not a device quantum-memory witness.

Experiment 3 preparation subsequently corrected `toggling_frame_features` to use the
column-state chronology `rotation @ frame` and transpose cumulative frames into the
toggling coordinates. A direct-propagation test now checks the convention. Current
OQE forecasts did not call this helper and were unaffected by the earlier issue.

## What could advance the field

A useful candidate contribution is a protocol that separates the predictive cost of
irreversible noise from the cost of retained classical or quantum information, under
declared control and calibration assumptions. The benchmark would report prediction,
model complexity, and what memory claims the observations actually support. Selected
additional controls would target ambiguity among models already consistent with the
data. This is a proposed direction, not a verified novelty claim or a completed
certification method.

In known-mechanism simulations, compare selected probes with random probes at equal
measurement budgets and a prespecified false-positive rate. Include coherent Markov,
classical correlated, and quantum-memory examples with confusable RB behavior, plus
realistic control and SPAM errors. Progress means fewer measurements to reliably
separate the declared alternatives, followed by a prospective independent data test.
Existing RB records cannot supply outcomes for interventions that were never run.

Unitary-only observations need not determine a unique process, but can sometimes
bound or witness temporal quantum correlations. Thus "RB can never witness memory"
would be too strong. The relevant next step is to test whether published witnesses
and assumptions apply to these observations, including control and SPAM uncertainty.
[White et al., Quantum 9, 1695 (2025)](https://quantum-journal.org/papers/q-2025-04-08-1695/)

Recent work derives RB blind spots under temporal correlations and conditional
criteria for quantum-memory witnesses. A model fitting an average or individual
survival probability well is not automatically satisfying one of those criteria.
[Srivastava et al., Physical Review Research 8, 023258 (2026)](https://arxiv.org/abs/2510.13051)

There is also established precedent for learning a dissipative Markovian embedding
of an effective environment. Reusing that architecture becomes a contribution only
with a new validated inference protocol, efficiency result, or experimental finding.
[Luchnikov et al., Physical Review Letters 124, 140502 (2020)](https://arxiv.org/abs/1902.07019)
