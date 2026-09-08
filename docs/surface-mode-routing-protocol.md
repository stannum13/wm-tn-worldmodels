# Persistent surface-code mode-routing protocol

Precommitted directional protocol, 8 September 2026. Outcomes from the confirmatory
stream were not inspected when this protocol was committed.

## Purpose

Test whether the synthetic `ACTIVATE_MODE` result survives contact with a standard
rotated surface-code circuit and a trusted MWPM decision layer. A persistent nuisance
mode switches reset and measurement flip probability between 0.001 and 0.01 while
after-Clifford depolarization remains 0.001. Each mode has a separately calibrated
Stim detector error model and PyMatching graph.

This benchmark changes a whole graph template rather than a local patch. It is the
simplest external-code test of causal graph-potential selection, not yet the intended
local overlay implementation.

## Fixed design

- Stim `surface_code:rotated_memory_z`, distances and rounds 3, 5, and 7.
- Persistent shot-level mode transition matrix `[[0.995, 0.005], [0.10, 0.90]]`.
- Independent calibration: 65,536 shots per mode and distance.
- Test: 512 independent streams x 512 shots per distance.
- Mode emission: Laplace-smoothed empirical distribution of total detector count.
- Memoryless router: stationary prior plus the current completed shot's detector count.
- Temporal router: the same emission and current observation plus the previous causal
  posterior and the fixed transition matrix.
- Router threshold 0.5; no threshold selection on test outcomes.
- Static baseline: nominal graph. Oracle: select the calibrated graph from true mode.
- Nominal-only null: same stream shape with every shot sampled from nominal mode.

The current shot's full syndrome is available before its decode, so routing is causal
at shot completion. It is not cycle-level early decoding and makes no sub-microsecond
latency claim.

## GO conditions

All LER changes use paired stream-level 95% t intervals.

1. **Graph-template relevance:** mode-oracle graph selection beats the static nominal
   graph with interval upper bound below zero at distances 3 and 5.
2. **Routing utility:** the temporal router recovers at least 80% of point-estimate
   oracle gain at distances 3 and 5.
3. **Temporal-state value:** temporal routing beats the matched memoryless router with
   interval upper bound below zero at two or more distances.
4. **Null safety:** on nominal-only streams, temporal routing versus the nominal graph
   has interval upper bound no greater than +0.01 percentage points at every distance.

Failure of condition 3 falsifies the need for temporal state for this observable and
mode separation; it does not falsify graph reweighting. Results at distance 7 are
reported but are not in the primary oracle/recovery conditions because the declared
sample size may contain too few logical failures.
