# Frozen decision-rule factor sensitivity sweep

Exploratory synthetic report, 8 September 2026.

## Design

The persistent-factor pilot used base pair-fault probability 0.03 and active-factor
probability 0.08. This follow-up freezes the selected HMM activation threshold and
EMA/hysteresis FSM separately for each observation-noise stratum, then evaluates a
3 x 4 grid of base probabilities {0.01, 0.03, 0.06} and active-factor probabilities
{0.02, 0.04, 0.08, 0.16} at observation sigma {0.5, 1.25}.

Each of 24 cells contains 512 independent episodes of 512 records (262,144 records).
There is no per-cell threshold or FSM retuning. However, the HMM and all decoder
likelihood tables—including the static and oracle comparators—are supplied the true
base/factor probabilities in each cell. This is therefore an exploratory sweep of
frozen *decision rules* under oracle-known per-cell noise parameters, not a fully
frozen deployment-transfer or learned-model test. Paired episode-level 95% intervals
for error-rate changes versus the static mixture are included in the artifact, but
the threshold counts below remain post-hoc point-estimate summaries.

## Result

At base probability 0.01 the mode-information opportunity is zero or only
0.0004--0.0050 percentage points in most cells. Gain-recovery ratios are consequently
unstable and those cells cannot rank architectures reliably.

For all eight cells with base probability 0.03 or 0.06 and sigma 0.5:

- HMM `ACTIVATE_MODE` recovers 93.2--98.2% of mode-information gain;
- HMM `FORK(K=2)` recovers 93.8--100.0%;
- the frozen EMA/hysteresis FSM recovers 87.9--96.5%;
- all three exceed the directional 80% threshold in 8/8 cells.

For the matched eight cells at sigma 1.25:

- ACTIVATE recovers 14.2--79.8% and passes 80% in 0/8 cells;
- FORK recovers 40.0--81.9% and passes in 1/8 cells;
- the frozen FSM ranges from -6.5% to 78.0% and passes in 0/8 cells.

The point estimates associate higher observation noise with weaker gain recovery
across this narrow grid. Local hypothesis lifting is often strongest, but does not
reliably close the mode-information gap. Because the two sigma arms use separately
selected controllers, different simulated trajectories, and oracle per-cell
calibration, this sweep does not causally isolate observation noise or establish a
general FORK advantage.

## Decision

- Low-noise decision rules: the observed point threshold is met across the selected
  informative cells; this is not a confirmatory GO.
- High-noise decision rules: the observed point threshold is missed in 8/8, 7/8, and
  8/8 selected cells for ACTIVATE, FORK, and FSM respectively; this is not a
  pre-registered program-level NO-GO.
- FSM portability: useful only in the well-observed regime; it can become worse than
  static at high ambiguity.
- Next step: do not enlarge K blindly. Test whether an added observation factor or a
  targeted diagnostic action improves mode identifiability at matched sensing cost.

This grid was run after the nominal cells were examined, and the exclusion of
near-zero-opportunity cells is a post-hoc interpretability rule. A future confirmatory
campaign must freeze the grid, opportunity floor, controller, decoder calibration,
common-random-number design, and interval procedure in advance. The decisive next
experiment must hold nominal filter and decoder parameters fixed while the generator
shifts, and compare that arm with an estimated-calibration arm.

Artifact: `results/factor_parameter_sweep.json`.
Runner: `scripts/run_factor_parameter_sweep.py`.
