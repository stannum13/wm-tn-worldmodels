# First graph-overlay compiler screen

Directional synthetic report, 8 September 2026.

## Purpose

This first controlled experiment asks whether observation-conditioned nuisance-state
inference can improve potentials supplied to a trusted repetition-code decision layer,
and whether a single compiled EMA plus piecewise-linear LUT retains that gain. It
covers only `REWEIGHT`. The HMM modes are generator/filter states; no distinct decoder
configuration is activated, so `ACTIVATE_MODE` remains untested. It also cannot test factor insertion, rewiring,
local graph forks, qLDPC region solving, added observations, or hardware partitioning.

The simulator uses paired fault trajectories, distance 3, four known nuisance modes,
512 time steps per independent episode, and 512 test episodes per arm (262,144
records). The trusted layer exactly selects between the two error chains consistent
with each repetition syndrome under the supplied per-site probabilities. EMA decay is
selected on separate validation episodes by downstream logical error.

This is a deliberately active synthetic positive control. It does not show that a
finite mode grammar describes hardware noise.

## Main result

| mode scope | emission sigma | static LER | memoryless-observation LER | oracle LER | known-parameter HMM LER | static-to-oracle gap closed | compiled EMA LER | HMM gain retained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local | 0.50 | 0.4601% | 0.4505% | 0.4440% | 0.4467% | 83.3% | 0.4593% | 5.7% |
| local | 1.25 | 0.4505% | 0.4494% | 0.4379% | 0.4494% | 9.1% | 0.4505% | 0.0% |

In the low-ambiguity local arm, the privileged HMM (which is given the true generator
transition, emissions, and error table) improves absolute logical error over static by
0.01335 percentage points; its paired episode 95% interval is [-0.02065, -0.00605]
points, with negative values favoring HMM. It descriptively closes 83.3% of the
static-to-oracle gap, although the relative LER improvement is only 2.90%, below the
proposed stronger 10% effect-size gate. The ratio has no uncertainty interval.

However, the memoryless current-observation model already closes 59.5% of that gap.
HMM improves only 10 net errors among 262,144 records beyond memoryless, -0.00382
percentage points with paired episode 95% interval [-0.00952, +0.00189]. The incremental
value of temporal state is therefore unresolved.

At sigma 1.25 the known-parameter HMM closes only 9.1% of the static-to-oracle gap and
its paired interval against static crosses zero. Because HMM versus memoryless is
already unresolved at sigma 0.5, this does not isolate a benefit from temporal state.
The current-time observation is assumed available before the decision.

The selected one-state EMA/LUT program is a clear compression NO-GO: it retains just
5.7% of the HMM gain in the favorable arm, far below the 80% gate. This rejects this
student, not the full finite grammar containing CUSUM, hysteresis, and explicit modes.

## Routing result

Mode entropy predicts individual events where the HMM lane repairs a
memoryless decision (AUROC 0.831 in the favorable local arm). At an exact retrospective
20% compute budget it captures 70.6% of 34 helpful events. However, syndrome density
has AUROC 0.934 and captures all 34 under the same budget. Ties in the discrete density
score use a stable index rule, so this is not a statistical superiority result. Mode
entropy nevertheless fails the predeclared simple-baseline advancement screen.

Helpful events are very sparse (11--103 across the four arms), and the budget ranking
is retrospective rather than an online threshold. No routing deployment claim is
supported.

## Benchmark failures exposed

The shared-mode arms have no measurable static-to-oracle opportunity under the
marginal-reweight interface on these sampled records. Marginal per-site probabilities
discard correlation induced by a shared latent mode, so this is not proof that the
generator has no shared-mode information. The arm cannot test whether global pooling
is useful and should be redirected to a correlated-factor decoder.

An earlier distance-5 pilot also produced almost no logical failures. It was discarded
before method interpretation and motivated the distance-3 positive-control setting.
These failures illustrate why every defect arm must first demonstrate a nonzero oracle
opportunity before architecture comparisons are meaningful.

## Decision

- observation-conditioned `REWEIGHT`: directional positive control in the observable
  local arm; temporal HMM state over the memoryless reweighter remains unresolved.
- single EMA/LUT compilation: NO-GO; next cheapest students are explicit four-mode FSM,
  HMM-to-threshold distillation, and CUSUM+hysteresis.
- entropy routing: fails the simple-baseline advancement screen on current development
  data; no statistical superiority between routers was established.
- shared/global state: unresolved because its marginal-reweight arm has zero sampled
  oracle opportunity; redirect the defect to a correlated-factor decoder.
- graph-overlay thesis: not decided. The remaining operations require distinct
  defect-isolating generators.

The primary statistical unit here is the independent simulated episode. Intervals are
paired across episodes. All architecture and generator choices remain development work
until the complete defect suite and thresholds are frozen.

Artifact: `results/adaptive_compiler_screen.json`.
Runner: `scripts/run_adaptive_compiler_screen.py`.
