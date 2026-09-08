# Conditional-risk training helps, but this grammar does not compress the teacher

8 September 2026. Ten prior distance-3 evidence models receive entirely fresh
action-fitting, selection and evaluation streams. Exact teacher probabilities are
used only to fit residual heads; selected students use cheap evidence and matching
energies at inference. Outcome-trained controls have identical feature and model
budgets, including the same six affine/spline/regularization candidates.

| Method | Mean logical error |
| --- | ---: |
| Previously selected static correlated matching | 2.04124% |
| Outcome-trained residual | 1.96083% |
| Conditional-risk-trained residual | 1.94481% |
| Exact causal teacher restricted to endpoint predictions | 1.81931% |
| Unrestricted exact causal teacher | 1.77589% |

Teacher-minus-outcome error is -0.01602 pp, with a 95% interval across ten model/
selection/evaluation replicates of [-0.02758, -0.00447]. Eight replicate point
estimates improve. This is a resolved directional improvement, but its upper bound
does not clear the predeclared -0.01-pp minimum-effect gate.

The selected student recovers only 11.3% of the outcome-trained-to-restricted-teacher
gap. The descriptive replicate-bootstrap interval is [5.0%, 17.1%], far below the
80% compression target. All bootstrap denominators were positive. Both scientific
GO conditions therefore fail; the numerical sign of the small improvement does
not override them.

The affine action has 67 inputs. Its spline alternative adds 20 hinge terms, giving
87 inputs. Both families select splines in eight of ten fits, but this selection
frequency alone does not demonstrate superiority over a matched affine-only
baseline. This is a small additive linear-spline model related to a simple KAN
primitive; no general KAN architecture or latency advantage is established.

This failure narrows the research direction. Less noisy teacher targets alone do
not make these action features sufficient. Even perfect selection between the two
endpoints leaves a further 0.04341-pp gap to the unrestricted exact teacher. A
future program needs more informative decision features, additional legitimate
candidate corrections, or graph factors representing some of the lost joint
structure. Merely increasing the number of affine/spline variants has little
support from this result.

This study uses 1,310,720 fresh evaluation shots and the same number separately
for action fitting and selection. Evidence models, known nominal regimes and exact
teacher tables are inherited; intervals do not represent a new end-to-end evidence
training campaign. The experiment covers nominal d3 only; transfer, null safety
and runtime of the new spline heads have not been established.

Protocol: [surface-teacher-distillation-protocol.md](surface-teacher-distillation-protocol.md).
Artifact: `results/surface_teacher_distillation.json`.
