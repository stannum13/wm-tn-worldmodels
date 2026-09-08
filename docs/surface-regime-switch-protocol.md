# Surface-code incompatible-regime protocol

Precommitted confirmatory protocol, 8 September 2026. The final seeds and outcomes
were not inspected when this document was committed. Noise levels and morphology
features were chosen in an explicitly exploratory pilot.

## Benchmark

Stim rotated-memory-Z circuits alternate between two persistent regimes:

- measurement/reset dominated: after-Clifford depolarization 0.001 and
  reset/measurement flips 0.02;
- gate/data dominated: after-Clifford depolarization 0.01 and
  reset/measurement flips 0.001.

The symmetric regime transition matrix is `[[0.98, 0.02], [0.02, 0.98]]`. Distances
and rounds are 3, 5, and 7. Each distance uses 512 independent streams of 512 shots.

## Strong fixed benchmark

On 65,536 independent calibration shots per regime, select the lowest mean-LER
PyMatching graph from a fixed 5 x 5 grid:

- after-Clifford depolarization {0.001, 0.003, 0.005, 0.007, 0.01};
- reset/measurement flip {0.001, 0.005, 0.01, 0.015, 0.02}.

This 25-way calibration search is the outcome-blind static benchmark. Test outcomes
cannot select it.

## Adaptive method

For each completed shot, compute total detector count, coincident detector pairs at
the same position in adjacent rounds, coincident spatial-neighbor pairs within a
round, and detector count per round. Fit one L2-regularized affine log-likelihood
ratio on the independent balanced calibration set. No hidden layer or decoder
labels enter the head.

Compare:

- the selected fixed graph;
- a memoryless affine router;
- the same affine evidence accumulated by a one-scalar causal Bayesian filter;
- true-regime selection between the two endpoint graphs.

The decision is available after a shot's detectors arrive. This is shot-causal, not
cycle-level early decoding.

## GO conditions

Paired stream-level 95% t intervals are authoritative.

1. **Benchmark win:** causal routing beats the selected 25-way static benchmark with
   interval upper bound below zero at distances 3 and 5.
2. **Temporal value:** causal routing beats its matched memoryless router with interval
   upper bound below zero at distances 3 and 5.
3. **Oracle recovery:** causal routing recovers at least 80% of the point-estimate gap
   from the selected fixed graph to true-regime endpoint selection at distances 3 and
   5.
4. **Scaling direction:** causal routing has lower point LER than the selected fixed
   graph at distance 7; statistical resolution is not required there.
5. **Stationary controls:** under each non-switching physical regime, the causal router
   is no worse than the graph calibrated for that regime by more than +0.02 pp at
   distances 3 and 5, using the upper interval bound.

Passing establishes a benchmark win for this declared synthetic circuit family, not
SOTA decoding, hardware advantage, or robustness to an unmodelled regime.
