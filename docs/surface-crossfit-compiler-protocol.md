# Cross-fitted graph-router compiler protocol

Precommitted confirmation, 8 September 2026. Final seed 20261231 was not inspected
when this protocol was committed. This protocol corrects the validation-reuse defect
identified in the residual-router result.

## Compiler

The circuit regimes, detailed local feature grammar, endpoint PyMatching graphs, and
25-way static benchmark are unchanged. Data roles are now disjoint:

- 65,536 independent shots per regime fit regime evidence and select the static graph;
- 256 x 512 persistent shots fit a residual action head only on decoder disagreements;
- a separate 256 x 512 persistent stream selects the deployed policy;
- 512 x 512 final shots evaluate it.

The residual receives both endpoint matching solution weights and their difference in
addition to detailed detector features and causal regime log-odds. These graph-native
energies expose each decoder's confidence without permitting a free-form correction.

The policy-selection set chooses the lowest-LER member of a fixed candidate family:

- the previous aggregate causal router at threshold 0.5;
- detailed causal posterior thresholds {0.1, ..., 0.9};
- the cross-fitted energy-residual router.

## GO conditions

1. **Benchmark win:** the compiled policy beats the calibration-selected 25-way
   static graph with paired stream-level 95% interval upper bound below zero at
   distances 3 and 5.
2. **Adaptive frontier:** it beats the prior aggregate causal router at distance 3
   with interval upper bound below zero and is no worse than +0.02 percentage points
   at distance 5.
3. **Opportunity recovery:** it recovers at least 50% of the static-to-mode-informed
   point gap at distance 3 and at least 80% at distance 5.
4. **Stationary safety:** under either stationary regime, its upper interval bound
   versus the corresponding endpoint graph is no greater than +0.05 pp at either
   distance.
5. **Selection integrity:** action-head fitting records and policy-selection records
   are disjoint and neither overlaps final test.

Passing establishes a reproducible win on this declared synthetic Stim/PyMatching
benchmark. It is not a claim of general decoder SOTA or hardware advantage.
