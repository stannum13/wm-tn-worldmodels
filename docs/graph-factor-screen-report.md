# Missing parity-factor screen

Directional synthetic report, 8 September 2026.

## Question

Can a bounded `INSERT_FACTOR` operation repair a topology defect which edge reweighting
cannot, and does activating that factor in the wrong regime cause measurable harm?

The tiny base model contains two disjoint pair faults, each with zero logical effect.
A correlated physical event flips the same four detectors as both pair faults together
but has logical effect one. Thus the syndrome alone admits two explanations with
different logical outcomes. The trusted layer is an exhaustively compiled
minimum-weight-error lookup over the supplied parity factors.

This model deliberately constructs the claimed defect. It is a unit-scale positive
control for the operation, not a realistic QEC threshold or hardware result.

## Result

Each arm contains 512 independent episodes of 1,024 records (524,288 records total).
All decoders see identical syndromes and labels.

| arm | base pair model | insert correct factor | insert wrong factor |
|---|---:|---:|---:|
| correlated factor present at p=0.04 | 4.0495% | 0.3212% | 4.0495% |
| correlated factor absent | 0.0000% | 0.0887% | 0.0000% |

With the factor present, insertion reduces logical error by 3.7283 percentage points,
or 92.1% relative. The paired episode 95% interval for the change is
[-3.7799, -3.6767] points. The wrong-support/logical-neutral motif does not change the
base predictions.

In the null arm, leaving the correct factor active changes 465 decisions and harms
logical error by 0.0887 points, interval [+0.0807, +0.0966]. This is below the
program's provisional 0.10-point stationary-harm cap in point estimate but is not a
safety argument: the result demonstrates that unconditional structural adaptation has
a real false-activation cost.

## Decision

- `INSERT_FACTOR`: positive-control GO. The correct structural operation closes the
  constructed missing-factor gap.
- unconditional insertion: NO-GO. The same operation harms the factor-absent regime.
- wrong motif: negative-control pass for this exact motif, not evidence that arbitrary
  wrong motifs are harmless.
- next experiment: introduce a causal, imperfect observation of factor activation and
  compare static-off, static-on, oracle gating, HMM/FSM gating, and local `FORK(K=2)`.
  Score gain retained, false activations, and stationary harm.

The crucial inference is architectural: reweighting a pair graph cannot express two
fault explanations with identical detector support and different logical effect. A
factor/hypothesis overlay can. Whether real device failures contain a small reusable
motif library remains an empirical question.

Artifact: `results/graph_factor_screen.json`.
Runner: `scripts/run_graph_factor_screen.py`.
