# Independent audit of the real-QEC branch

This audit was requested after the stability-9 soft, feature, Markov, and matching
experiments. It was performed by an independent review agent and is recorded here to
make the decision boundary explicit.

## What the evidence supports

- The real-data experiments are credible negative or baseline studies. They do not
  yet demonstrate a learned world model, closed-loop control, real-time deployment,
  or a state-of-the-art decoder.
- The soft replay only rejects the tested *independent marginal soft-parity*
  construction. It does not show that analog information is useless for QEC.
- The static-feature and categorical-Markov failures reject narrow model families;
  they do not show that spatial or temporal dependence is useless.
- The three selected circuit-noise probabilities are effective matching weights under
  a misspecified model, not identified hardware error rates.
- All QEC results are offline logical replay. Host timings do not establish FPGA or
  feedback-controller latency.

## Statistical correction

Early QEC artifacts accidentally grouped uncertainty blocks by logical label. Those
intervals were invalid for acquisition drift. The affected experiments were rerun
using contiguous HDF5 row-order blocks. Point estimates were unchanged, but all
reports now use the corrected intervals. Since the public files omit per-shot
timestamps, this protects local dependence in file order without claiming verified
wall-clock chronology.

## Strongest next falsification

The current uniform circuit-noise PyMatching control reproduces the released
stability-9 scale, but it is not the strongest released baseline. The released
pairwise-correlation graph leaves material structural headroom on the deepest
with-resets circuit. The next validity gate is therefore:

1. Reproduce the published pairwise-correlation detector graph on development data.
2. Match its released deepest-circuit error within 0.20 percentage points, after
   resolving the public 23-versus-24-round naming convention.
3. Freeze graph construction, regularization, update cadence, and all thresholds.
4. Compare a causal slow graph-weight estimator against the strongest static graph;
   keep PyMatching in the hot path.
5. Confirm only once on the untouched `stability_8_without_resets` acquisition, with
   a label-free syndrome calibration prefix and locked evaluation suffix.

The research GO condition is at least 1% relative improvement at the maximum-round
endpoint, paired row-order-block lower confidence bound above zero, no greater than
0.20 percentage-point harm at any shallower depth, and no more than 10% hot-path p99
latency regression. If the reproduction gate fails, debug the baseline. If it passes
and the adaptive model misses the research gate, stop this branch rather than adding
capacity.

## Architectural implication

Particle or nonlinear estimation belongs in the slow lane only when the latent graph
state is demonstrably multimodal or nonlinear. A static sparse estimator and an EKF
are required controls. A particle method earns inclusion only by improving held-out
logical error or calibration under the same update budget; posterior complexity by
itself is not a contribution.
