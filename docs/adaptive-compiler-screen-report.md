# First graph-overlay compiler screen

Directional synthetic report, 8 September 2026.

## Purpose

This first controlled experiment asks whether causal nuisance-state inference can
improve potentials supplied to a trusted repetition-code decision layer, and whether a
single compiled EMA plus piecewise-linear LUT retains that gain. It covers only the
`REWEIGHT` and `ACTIVATE_MODE` categories. It cannot test factor insertion, rewiring,
local graph forks, qLDPC region solving, added observations, or hardware partitioning.

The simulator uses paired fault trajectories, distance 3, four known nuisance modes,
512 time steps per independent episode, and 512 test episodes per arm (262,144
records). The trusted layer exactly selects between the two error chains consistent
with each repetition syndrome under the supplied per-site probabilities. EMA decay is
selected on separate validation episodes by downstream logical error.

This is a deliberately active synthetic positive control. It does not show that a
finite mode grammar describes hardware noise.

## Main result

| mode scope | emission sigma | static LER | oracle LER | causal local HMM LER | oracle gap closed | compiled EMA LER | teacher gain retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 0.50 | 0.4601% | 0.4440% | 0.4467% | 83.3% | 0.4593% | 5.7% |
| local | 1.25 | 0.4505% | 0.4379% | 0.4494% | 9.1% | 0.4505% | 0.0% |

In the low-ambiguity local arm, the HMM improves absolute logical error over static by
0.01335 percentage points; its paired episode 95% interval is [-0.02065, -0.00605]
points. It passes the directional >=80% oracle-gap recovery rule, although the relative
LER improvement is only 2.90%, below the proposed stronger 10% effect-size gate.

The result collapses as observations become noisier. At sigma 1.25 the local filter
closes only 9.1% of the oracle gap and its paired interval crosses zero. Thus causal
state helps only when the latent mode is sufficiently observable in this screen.

The selected one-state EMA/LUT program is a clear compression NO-GO: it retains just
5.7% of the HMM gain in the favorable arm, far below the 80% gate. This rejects this
student, not the full finite grammar containing CUSUM, hysteresis, and explicit modes.

## Routing result

Mode entropy predicts individual events where the causal slow lane repairs a
memoryless decision (AUROC 0.831 in the favorable local arm). At an exact retrospective
20% compute budget it captures 70.6% of 34 helpful events. However, syndrome density
has AUROC 0.934 and captures all 34 under the same budget. The ambiguity hypothesis
therefore fails its required simple-baseline comparison in this arm.

Helpful events are very sparse (11--103 across the four arms), and the budget ranking
is retrospective rather than an online threshold. No routing deployment claim is
supported.

## Benchmark failures exposed

The shared-mode arms have no measurable static-to-oracle opportunity: changing their
probabilities does not change the trusted layer's decision on these generated records.
They therefore cannot test whether global pooling is useful. This arm is a benchmark
NO-GO and must be replaced by a generator with a prespecified shared defect whose
optimal graph operation actually differs from the static graph.

An earlier distance-5 pilot also produced almost no logical failures. It was discarded
before method interpretation and motivated the distance-3 positive-control setting.
These failures illustrate why every defect arm must first demonstrate a nonzero oracle
opportunity before architecture comparisons are meaningful.

## Decision

- `REWEIGHT`/local mode state: directional GO only in the observable local arm; retain
  as the O1/O2 positive control.
- single EMA/LUT compilation: NO-GO; next cheapest students are explicit four-mode FSM,
  HMM-to-threshold distillation, and CUSUM+hysteresis.
- entropy routing: NO-GO against syndrome density on current data.
- shared/global state: unresolved because its arm has zero oracle opportunity; redesign
  the defect, do not tune the estimator.
- graph-overlay thesis: not decided. The remaining operations require distinct
  defect-isolating generators.

The primary statistical unit here is the independent simulated episode. Intervals are
paired across episodes. All architecture and generator choices remain development work
until the complete defect suite and thresholds are frozen.

Artifact: `results/adaptive_compiler_screen.json`.
Runner: `scripts/run_adaptive_compiler_screen.py`.
