# Frozen graph-control deployment shift

Precommitted synthetic directional result, 8 September 2026.

## Outcome

The soft local hypothesis lift, `FORK(K=2)`, is the only frozen operation that meets
the precommitted safety condition in all 24 cells. Frozen hard `ACTIVATE_MODE` and the
compiled FSM each fail in the sparse, high-ambiguity cell (base probability 0.01,
active-factor probability 0.16, sigma 1.25).

| Frozen operation | cells within +0.05 pp upper-bound cap | worst upper bound |
| --- | ---: | ---: |
| ACTIVATE_MODE | 23/24 | +0.0562 pp |
| FORK(K=2) | 24/24 | +0.0344 pp |
| compiled FSM | 23/24 | +0.0686 pp |

The formal FORK safety GO must not be read as no-harm evidence. In its worst cell,
frozen FORK raises LER from 0.0240% to 0.0504%, a +0.0263 percentage-point change
with paired 95% interval [+0.0183, +0.0344] pp. The operation is detectably harmful
but remains below a tolerance that was too permissive to encode “safe.” Future gates
should require a non-inferiority margin justified against the baseline LER scale.

## Utility under rate shift

Using the protocol's fixed 0.02 pp opportunity floor and excluding the two nominal
rate cells:

- frozen ACTIVATE recovers at least 70% of the calibrated-oracle HMM opportunity in
  13/13 informative cells (point-estimate range 99.7--103.4%);
- frozen FORK does so in 14/14 informative cells (range 75.0--100.0%).

Both pass the precommitted 75% utility condition. This does not mean their
probability estimates remain calibrated. It means that, once a cell has enough
logical-error opportunity to measure, the relevant decisions are unusually
insensitive to these rate shifts in this small parity problem. The low-base corner
shows why an opportunity-filtered utility metric must be paired with an all-cell
safety test.

## Can calibration stay outside the hot path?

The held-out estimator identifies the exact grid cell in 24/24 cases from 65,536
unlabelled records per cell. Estimated and oracle-calibrated predictions are therefore
identical in this experiment. Only frozen FORK has cells meeting the protocol's
minimum 0.02 pp recalibration opportunity: estimated calibration recovers 100% in
4/4 such cells, passing its directional condition. ACTIVATE and FSM have no eligible
cells, so no calibration-recovery conclusion is available for them.

This is a deliberately easy finite-grid identification result. It establishes a
mechanism: a slow-path estimate of two nuisance rates can repair a fixed hot-path
graph program without retraining or changing its state machine. It does not establish
continuous-parameter sample efficiency, unsupervised identifiability outside this
model family, or robustness to transition, observation, and topology shifts.

## Scientific decision

- **Retain FORK(K=2)** as the default bounded operation for rate uncertainty: it is
  softer and more robust than threshold activation, and it passes both formal safety
  and utility conditions here.
- **Do not claim no harm.** Add an abstention or evidence gate before deployment in
  low-event-rate regimes; the present fixed tolerance hid a twofold relative LER rise.
- **Keep calibration off the hot path.** The exact-grid result justifies testing a
  periodic two-parameter estimator with smaller windows and continuous rates.
- **Do not enlarge hypothesis width yet.** These failures are calibration/evidence
  problems, not evidence that K=2 lacks representational capacity.

The decisive follow-up is a calibration sample-complexity curve under continuous
off-grid rate shifts, followed by transition-dynamics and observation-mean shifts.
That experiment should use a relative as well as absolute non-inferiority margin.

Protocol: [factor-deployment-shift-protocol.md](factor-deployment-shift-protocol.md).
Machine-readable artifact: `results/factor_deployment_shift.json`.
