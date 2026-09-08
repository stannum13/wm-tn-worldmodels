# Persistent-factor gating and local hypothesis lift

Directional synthetic report, 8 September 2026.

## Question

When a known correlated factor switches between background and high-rate modes, does
causal state improve a compiled graph-template selector, and does retaining two local
hypotheses outperform hard mode selection under ambiguous observations?

The model has two ordinary pair faults and one four-detector logical factor. A binary
Markov mode changes the factor rate between 0.0001 and 0.08. A noisy scalar observation
is available before each decision. The privileged HMM uses the true transition,
emission, and fault parameters; its carried state incorporates past observations and
past syndromes. The current syndrome is used once at the current decision.

`ACTIVATE_MODE` selects one of two compiled factor-rate likelihood tables. Semantic
`FORK(K=2)` retains both tables in the four-detector region and marginalizes their
logical likelihoods. This is exact for the toy model, not an implementation of graph
copying or a hardware latency result.

Memoryless and HMM activation thresholds, and 95 EMA/hysteresis candidates, are
selected on independent validation episodes by logical error. Each test arm contains
512 independent episodes of 1,024 records (524,288 records).

## Result

Logical-error rates are percentages:

| observation sigma | static mixture | memoryless ACTIVATE | HMM ACTIVATE | memoryless FORK | HMM FORK | compiled FSM | mode-information oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.11196 | 0.04234 | 0.03738 | 0.04368 | 0.03738 | 0.03815 | 0.03548 |
| 1.25 | 0.10262 | 0.08373 | 0.07496 | 0.08316 | 0.07095 | 0.07896 | 0.03815 |

The oracle knows the latent rate mode, not the actual faults or logical outcome.

Temporal state has resolved incremental value in this constructed model. HMM versus
memoryless ACTIVATE changes LER by -0.00496 points at sigma 0.5, paired episode 95%
interval [-0.00729, -0.00263], and -0.00877 points at sigma 1.25, interval
[-0.01577, -0.00178]. For matched `FORK(K=2)`, temporal state changes LER by -0.00629
points, interval [-0.00902, -0.00357], and -0.01221 points, interval
[-0.01821, -0.00621], respectively.

Hypothesis width is conditional. At sigma 0.5, HMM FORK and HMM ACTIVATE make identical
decisions. At sigma 1.25, FORK reduces LER by 0.00401 points relative to ACTIVATE,
interval [-0.00740, -0.00061]. This is only 21 net errors in 524,288 records after the
joint-filter correction: statistically resolved in the simulated episode population,
but small and specific to this hand-constructed ambiguity.

The selected EMA/hysteresis FSM retains 99.0% and 85.5% of the HMM-ACTIVATE gain over
the static mixture. It passes the directional >=80% compiler rule for that same
operation. It retains 99.0% but only 74.7% of HMM-FORK gain, so it fails to compile the
richer high-ambiguity operation. At sigma 1.25 it is resolved worse than HMM ACTIVATE
by +0.00401 points, interval [+0.00082, +0.00719].

Neither HMM operation recovers 80% of the mode-oracle gap at sigma 1.25: ACTIVATE
recovers 42.9% and FORK 49.1%. The full primitive therefore does not pass an
across-ambiguity oracle-recovery gate.

## Frozen high-rate-factor-absent null

The selectors chosen in the active validation arm were applied unchanged when both
latent modes had only the 0.0001 background factor rate. Relative to the always-off
template, the FSM causes +0.00763 points of harm at sigma 0.5 (interval
[+0.00501, +0.01025]) and +0.01602 points at sigma 1.25 (interval
[+0.01208, +0.01996]). HMM FORK harms by +0.00534 points at sigma 0.5
(interval [+0.00301, +0.00767]) and +0.02193 points at sigma 1.25
(interval [+0.01738, +0.02648]). All are below the provisional +0.10-point cap, but
the costs are resolved and demonstrate that the controller must detect model mismatch
or fail closed. The latent label called “mode” is irrelevant in this null, so the
mode-information oracle is not interpreted.

## Decision

- Known-parameter temporal belief: positive-control GO over matched memoryless
  inference for this persistent-factor generator.
- `ACTIVATE_MODE`: semantic positive-control GO. The selected FSM directionally
  compiles at least 80% of the HMM-ACTIVATE gain in both tested ambiguity arms.
- `FORK(K=2)`: no added value at low noise; small resolved value at high noise.
- FSM as a compiler of FORK: NO-GO at high noise (74.7% gain retention).
- Across-ambiguity archive admission: NO-GO because neither HMM operation closes 80%
  of the mode-oracle gap at sigma 1.25.

The reported recovery ratios are descriptive and have no bootstrap interval. The
finite two-mode grammar, factor support, emissions, and probabilities are known by
construction. No model was learned, no runtime or resource use was measured, and the
stationary mixture makes the same decisions as always-on in these arms. The adaptive
gain is therefore primarily correct factor veto/deactivation, complementing—but not
repeating—the earlier missing-factor insertion positive control.

Artifact: `results/factor_gating_screen.json`.
Runner: `scripts/run_factor_gating_screen.py`.
