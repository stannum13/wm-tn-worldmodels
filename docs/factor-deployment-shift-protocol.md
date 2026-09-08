# Factor deployment-shift protocol

Precommitted directional protocol, 8 September 2026. Outcomes were not inspected
when this protocol was committed.

## Question

Do graph-operation decision rules selected at the nominal synthetic condition remain
useful when base and correlated-factor rates shift but the filter and decoder
likelihoods are genuinely frozen? If not, can a small held-out calibration window
recover the loss without changing the graph-operation program?

## Design

- Nominal calibration: base probability 0.03, active-factor probability 0.08,
  inactive-factor probability 0.0001.
- Deployment grid: base probability {0.01, 0.03, 0.06} crossed with active-factor
  probability {0.02, 0.04, 0.08, 0.16} and observation sigma {0.5, 1.25}.
- Each test cell: 512 independent episodes of 512 records.
- Each estimated-calibration cell: a disjoint 128 x 512-record calibration sample.
- A common generator seed is reused across grid cells. Thresholding the same latent
  uniforms produces coupled trajectories and common observation noise, while all
  within-cell comparisons are paired episode by episode.
- HMM activation thresholds and FSM rules remain exactly those selected in the
  nominal pilot. No operation rule is selected on these outcomes.

The frozen arm assumes nominal base/factor rates in both its causal HMM and decoder
likelihoods. The calibrated-oracle arm uses true cell rates and is an upper comparator,
not deployable evidence. The estimated arm selects base and active-factor rates from
the fixed deployment grid by likelihood on held-out syndromes, using an
observation-only causal mode posterior; it never accesses true modes or logical labels.

## Fixed metrics and conditions

All logical-error-rate (LER) changes use paired episode-level 95% t intervals.
An informative cell has calibrated-oracle HMM improvement over frozen static of at
least 0.02 percentage points; this floor is fixed before running.

1. **Frozen safety GO:** the upper interval bound for frozen ACTIVATE and frozen FORK
   versus frozen static is no greater than +0.05 percentage points in every cell.
2. **Frozen utility GO:** in at least 75% of informative non-nominal cells, a frozen
   operation recovers at least 70% of the calibrated-oracle HMM improvement over
   frozen static.
3. **Calibration GO:** in at least 75% of cells where oracle recalibration improves
   the corresponding frozen operation by at least 0.02 percentage points, estimated
   calibration recovers at least 80% of that improvement.
4. **Compilation safety GO:** the upper interval bound for the frozen FSM versus
   frozen static is no greater than +0.05 percentage points in every cell.

These are program-level directional conditions for this synthetic mechanism only.
They do not establish hardware latency, surface-code LER, or robustness to shifts in
transition dynamics, observation means, topology, or unmodelled error mechanisms.
