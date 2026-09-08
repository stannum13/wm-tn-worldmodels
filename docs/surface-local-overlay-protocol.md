# Local surface-code overlay protocol

Precommitted directional protocol, 8 September 2026. Confirmatory outcomes were not
inspected when this protocol was committed.

## Question

Does spatially local graph adaptation outperform a global noise-mode template when a
persistent reset/readout burst affects only the left half of surface-code ancillas?
Does a local detector statistic outperform global syndrome density for routing it?

## Fixed design

- Stim rotated-memory-Z circuits with distance=rounds in {3, 5, 7},
  after-Clifford depolarization 0.001, and reset/measurement flips 0.001.
- The burst composes an extra Pauli-X channel onto reset/readout error instructions
  for ancillas whose x coordinate is no greater than the code distance. The resulting
  flip probability is 0.01 only in that half.
- A deliberately wrong global template applies the burst to every ancilla.
- Shot-level transition matrix `[[0.995, 0.005], [0.10, 0.90]]`.
- Independent calibration: 65,536 shots per physical mode and distance.
- Test: 512 streams x 512 shots per distance; fixed posterior threshold 0.5.
- Local observation: fired detectors with x coordinate no greater than the distance.
  Global observation: total fired detectors.
- Every decision uses a fixed PyMatching graph selected after the completed shot.

## GO conditions

Paired stream-level 95% t intervals are authoritative. Distances 3 and 5 are primary.

1. **Local overlay relevance:** true-mode local-graph selection beats static nominal
   decoding with interval upper bound below zero at both primary distances.
2. **Spatial specificity:** true-mode local-graph selection beats true-mode selection
   of the wrong global graph with interval upper bound below zero at either primary
   distance and is not worse by more than +0.01 percentage points at the other.
3. **Local observability:** local-count temporal routing beats total-count temporal
   routing under the same criterion as condition 2.
4. **Temporal value:** local-count temporal routing beats its matched memoryless
   router with interval upper bound below zero at two or more distances.
5. **Benchmark win:** local-count temporal routing beats the strongest outcome-blind
   fixed graph (nominal, local-burst, or global-burst, selected on independent
   calibration) with interval upper bound below zero at both primary distances.
6. **Null safety:** local temporal routing is no worse than +0.01 pp by upper interval
   bound on nominal-only streams at every distance.

The fixed-graph benchmark is selected solely by calibration LER, not test outcomes.
This remains a synthetic spatial positive control. Full precompiled PyMatching
objects realize the templates, so no sparse in-place update latency is claimed.
