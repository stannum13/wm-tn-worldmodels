# Real benchmark report: correlated-noise randomized benchmarking

## Benchmark and protocol

I used the raw `standard_rb_1q_full_data.json` records from the public
[`guochu/pt_recovery`](https://github.com/guochu/pt_recovery) release. Each record
contains a Clifford-action string (`cl_ops`, alphabet size 24) and measured survival
probability `p0`. The processed repository `data.json` was not used because its split
is a random partition within each sequence length; that is unsuitable for a long-
horizon extrapolation claim.

The primary split uses all 3,800 sequences with lengths 2–20 for training and all
4,000 sequences with lengths 21–40 for testing. The test sequences are disjoint, and
no target values enter the features. Models are:

* length-only ridge regression (2 parameters),
* standard RB exponential decay (3 parameters),
* length plus unordered Clifford counts (26 parameters),
* length plus ordered adjacent-pair counts (602 parameters),
* a small GRU (1,457 parameters) trained on the same short sequences.

The metric is RMSE of held-out survival probability. This tests extrapolation, not
interpolation. It is deliberately weaker than a process-tensor reconstruction and
therefore should not be presented as a reproduction of the paper's OQE model.

## Results at bias amplitude 0.1

| model | parameters | test RMSE |
|---|---:|---:|
| RB exponential | 3 | **0.04522** |
| length-only | 2 | 0.05266 |
| unordered action counts | 26 | 0.05237 |
| ordered adjacent pairs | 602 | 0.05546 |
| GRU | 1,457 | 0.07455 |

The order-sensitive baselines and GRU do not outperform the RB exponential baseline on
this split. This is a useful negative result: a generic “memory” feature is not
evidence of a useful process memory, and parameter count does not buy extrapolation.

## Bias sweep

The same length extrapolation was run across the available bias amplitudes. Standard
RB decay is stronger than the linear and action-count baselines. At intermediate
biases its errors become much larger because individual sequences have wide outcome
variation,
which is a strong distribution-shift warning rather than evidence for a tensor-network
advantage.

## Relation to the published work

The paper [*Learning and forecasting open quantum dynamics with correlated noise*](https://www.nature.com/articles/s42005-025-01944-2)
reports a physics-inspired open-quantum-evolution reconstruction from randomized
benchmarking data and emphasizes forecasting beyond the training time range. Its
claim is stronger and structurally different from the baselines here: it reconstructs
an explicit system–memory model/process tensor. The present benchmark therefore does
not claim to beat that paper. It establishes the bar that a new process-MPO or OQE
implementation must clear: beat the length-only baseline on the same raw records,
retain validity, and improve under held-out bias—not merely fit interpolated points.

## Go/no-go and next experiment

The standalone learned baselines fail the intended claim, so the next useful experiment is not a
larger GRU. It is a faithful OQE/process-tensor reproduction on one bias setting,
with an independently checked process-tensor reference such as
[OQuPy](https://github.com/tempoCollaboration/OQuPy),
followed by a held-out-bias test. Report survival probability, multi-time observables,
memory size, wall time, and invalid-state/causality residuals together. A new method
should be called an improvement only if it beats the published-style OQE baseline on
at least one extrapolative split at matched data and compute budgets.

Subsequent analysis of the released OQE forecast artifacts found that a validation-
gated residual hybrid can outperform both RB decay and released OQE predictions on
lengths 41–60. See [`architecture-findings.md`](architecture-findings.md). This is a
narrow prediction result, not a reproduction of every claim in the paper.
