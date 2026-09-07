# Architectural findings from the released OQE forecasts

Review update (2026-09-06): the forecasting gains below remain exploratory. The
independent [scientific review](scientific-review.md) identifies unresolved control
gauge, baseline capacity, and evaluation-protocol issues. New fixed-generator D1
depolarization/readout diagnostics improve several active-bias forecasts further,
without establishing environmental memory or identifying device noise parameters.

## The decisive baseline correction

The original linear length baseline was scientifically too weak. Standard randomized
benchmarking uses `A * alpha^length + B`; after implementing that form, it became the
strongest simple predictor. On the released length-41–60 forecasts, this three-
parameter model frequently beats the much larger OQE reconstruction.

This changes the target. A sequence model must explain deviations from RB decay, not
the decay itself.

## Where sequence-dependent signal exists

The released D6 OQE correction relative to RB has almost no correlation with observed
residuals at weak biases 0.1–0.2 or strong biases above roughly 0.58. It is predictive
in the intermediate 0.4–0.54 window. That window is also where sequence-to-sequence
variance is largest and the RB-only model fails most strongly.

The architecture should therefore be conditional: use a low-variance RB expert when
the sequence-specific signal is unsupported, and a structured dynamics expert when
validation data show transferable residual structure.

## Validation-gated residual shrinkage

For validation lengths 21–40, define the released OQE correction

`c = p_OQE - p_RB`

and fit one scalar

`w = clip(dot(c, y - p_RB) / dot(c, c), 0, 1)`.

The forecast is `p_RB + w*c`. A conservative gate sets `w=0` when the validation
estimate is below 0.5. Forecast lengths 41–60 are not used to fit `w`.

Across 14 biases, mean per-bias RMSE is:

| OQE dimension | released OQE | RB decay | gated hybrid |
|---:|---:|---:|---:|
| 1 | 0.2291 | 0.0911 | **0.0734** |
| 2 | 0.1835 | 0.0911 | **0.0743** |
| 3 | 0.1658 | 0.0911 | **0.0768** |
| 4 | 0.1499 | 0.0911 | **0.0769** |
| 5 | 0.1414 | 0.0911 | **0.0775** |
| 6 | 0.1339 | 0.0911 | **0.0770** |

For D1 the gate activates only at biases 0.4, 0.5, 0.52, and 0.54. In that window the
hybrid beats both components. At the other biases it exactly falls back to RB.

## Interpretation

The released OQE often gets a useful *direction* for sequence-specific corrections
but an unreliable amplitude. Shrinkage fixes calibration while preserving the
structured signal. More OQE memory improves the standalone OQE forecast, yet the
best hybrid uses D1. Thus this benchmark currently supports a small coherent
gate-dependent correction, not a claim that larger environmental memory is required.

This is the main design principle discovered so far:

> Decouple low-variance mean dynamics from high-variance structured residual dynamics,
> and activate the latter only when its residual predictability transfers across
> validation horizons.

## Independent reconstruction from the raw controls

The release references a `cliffords.json` file that is not present in the repository.
The RB sequences themselves recover it algebraically. Every sequence ends in its exact
inverse: length-2 sequences identify all 24 inverses, and length-3 sequences identify
575 of 576 binary products across bias runs. Inverse symmetry and associativity fix
the final product. The resulting table is isomorphic to the 24 proper signed-
permutation rotations of the Bloch sphere. Applying the transpose convention makes
every released sequence compose to identity in its listed execution order.

This recovers a consistent abstract control representation, not its unique alignment
with laboratory preparation and measurement axes. A global conjugation is a harmless
gauge transformation only when controls, dynamics, initial state, and measurement all
transform together; the current implementation fixes preparation and measurement to
`|0>`.

Using these reconstructed Clifford matrices, a differentiable repeated-unitary OQE
was trained independently. Training and validation use disjoint sequences spanning
lengths 2–40 from the released random split; forecasting uses lengths 41–60. This is
different from the initial generic-model experiment trained only on lengths 2–20.
Its Hermitian generator stores `(2D)^2` real parameters, including one unobservable
identity/global-phase direction. Removing the redundant complex parameterization
eliminated the failed D2 optimization seed in these runs, but did not remove every
physical parameter redundancy.

Length-41–60 results from three independent training seeds are:

| bias | RB RMSE | reconstructed D1 OQE | D1 gated hybrid |
|---:|---:|---:|---:|
| 0.40 | 0.1407 | 0.1491 | **0.0909** |
| 0.50 | 0.1834 | 0.1486 | **0.0938** |
| 0.52 | 0.1639 | 0.1700 | **0.1013** |
| 0.54 | 0.1294 | 0.2076 | **0.1163** |

At bias 0.5, D2 improves standalone OQE RMSE from 0.1486 to 0.1304, but worsens the
calibrated hybrid from 0.0938 to 0.1006. The extra memory is useful for raw fit yet
does not improve the best forecast. At inactive biases 0.1 and 0.6, validation gating
selects zero OQE weight and exactly recovers the stronger RB prediction.

Generic primitives did not beat RB in their own experiment. Across ranks 2, 4, and 8
and three seeds, additive oscillator memories and action-conditioned matrix-product
recurrences worsen the bias-0.5 RB forecast. They see Clifford IDs as tokens.
Because these runs used a different training/forecast horizon from reconstructed
OQE, their failure is not a controlled ablation establishing that the Clifford
representation or physical composition caused the performance gap. A matched-data,
matched-horizon comparison is still required. Residual calibration does have a direct
same-prediction comparison in the released-artifact results above.

## Scientific limits

The 0.5 gating threshold was formed during exploratory analysis on biases 0.1–0.61.
Biases 0.62–0.64 were then held as confirmation settings; all three correctly selected
the RB fallback and substantially beat released OQE forecasts. They do not confirm an
active hybrid because none lies in the intermediate-signal window. A second dataset
or idle-duration condition must confirm active mixing before this is presented as a
general improvement.

The result outperforms the released OQE prediction artifacts on this specific survival-
probability RMSE protocol. It does not outperform the paper's broader process-tensor
claims, multi-time characterization, or non-Markovianity measures.

## Next architectural step

The scientific review supersedes the earlier recommendation to prioritize a more
complex activation gate. First resolve the control/SPAM gauge and compare damped D1
and a general memoryless qubit CPTP channel. Then compare retained, dephased, and reset
environment models under a common protocol, with classical correlated noise as an
additional comparator. Freeze model selection before confirming on `idle180`; use
intervention-sensitive evidence before making claims about necessary quantum memory.
