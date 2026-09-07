# Process critique and smallest useful next step

## What is good

The plan identifies the right scientific object: an action-conditioned predictive
state, rather than a static density-matrix regressor. It also correctly warns against
random time-point splits and distinguishes experimental data from HEOM/FCI teacher
data.

## What is not yet a study

The repository contains only a plan and README; the promised loaders, splits, four
baselines, checks, and evaluation artifacts do not exist yet. The plan jumps from a
broad survey (process tensors, HEOM, N-representability, HOTRG, chemistry, and 1D/2D
dynamics) to a difficult real-device comparison without first proving that the data
and evaluation path work.

Specific process risks:

1. **Scope is not falsifiable as written.** Five hypotheses and five experiment
   tracks make it easy to select a favorable result after the fact. Choose one claim,
   one primary split, and one primary metric before implementation.
2. **No operational data contract.** “Recovered process tensors” and “multi-time
   tomography” need documented tensor shape, intervention encoding, uncertainty,
   missing-data policy, and units. Otherwise model differences can be parser
   differences.
3. **No statistical design.** The plan names held-out biases and horizons but not
   independent sequence counts, seed policy, confidence intervals, or paired tests.
4. **Constraint claims are underspecified.** CPTP residual, causality, and
   N-representability need equations and tolerances. Post-hoc projection must be an
   explicit ablation, with parameter count and compute measured for every model.
5. **The first experiment has too many moving parts.** Process-MPO, tomography
   reconstruction, control-family split, and long rollout are each a failure mode.
6. **Negative controls are missing.** Include a no-memory generator, an oracle, and a
   shuffled-control test; otherwise a gain can be leakage or extra parameters.

## Minimal directional experiments

`scripts/run_directional_experiments.py` runs two CPU-only controls with ten seeds:

* **Memory direction:** a scalar controlled process has known one-step action memory.
  A Markov linear predictor is compared with the same model plus one lagged action,
  using recursive 40-step rollouts on held-out sequences.
* **Physicality direction:** a damped qubit Bloch process is trained in-distribution
  and rolled out under stronger controls. Raw linear predictions can leave the Bloch
  ball; radial projection is scored for validity and trace distance.

These controls test whether the implementation recovers known effects. They are not
evidence about a real device; port the exact split and reporting protocol to public
tomography data only after they pass.

The runner also emits a no-memory null control and stores per-seed results plus
5th–95th percentile intervals when `--output` is supplied. Those intervals describe
seed variability, not a substitute for uncertainty over experimental sequences.
