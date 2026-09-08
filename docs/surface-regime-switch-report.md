# Incompatible surface-code regime routing

Precommitted confirmatory result, 8 September 2026.

## Outcome

The one-scalar causal router beats a 25-way calibration-selected static MWPM graph at
distances 5 and 7, and decisively beats the same affine head without memory at all
three distances. The full precommitted program is nevertheless a NO-GO because the
distance-3 benchmark interval crosses zero, distance-3 oracle recovery is only 21.5%,
and its stationary safety controls miss the declared margin.

| distance | selected static LER | causal LER | change vs static (95% CI, pp) | change vs memoryless (95% CI, pp) | mode-informed recovery |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2.1057% | 2.0782% | -0.0275 [-0.0605, +0.0056] | -0.1240 [-0.1487, -0.0992] | 21.5% |
| 5 | 1.9333% | 1.7700% | -0.1633 [-0.1968, -0.1298] | -0.0729 [-0.0882, -0.0575] | 85.8% |
| 7 | 1.6071% | 1.4912% | -0.1160 [-0.1495, -0.0824] | -0.0149 [-0.0224, -0.0074] | 96.2% |

Each distance used 262,144 shots. The fixed benchmark was selected on independent
balanced calibration data from a 5 x 5 grid of circuit-noise models. The adaptive
head used only syndrome morphology and one recurrent posterior scalar.

## Precommitted decisions

- **Benchmark win: NO-GO as written.** Distance 5 wins decisively; distance 3 is
  directionally favorable but unresolved.
- **Temporal value: GO.** The causal state beats the matched memoryless head at both
  primary distances and also at distance 7.
- **Oracle recovery: NO-GO.** Distance 5 passes at 85.8%; distance 3 reaches 21.5%.
- **Scaling direction: GO.** Distance-7 LER improves by 0.1160 pp with a resolved
  interval, exceeding the required directional result.
- **Stationary safety: NO-GO.** At distance 3, upper interval bounds are +0.0459 and
  +0.0468 pp versus the matching endpoint graph, above the +0.02 pp margin. Distances
  5 and 7 satisfy the margin.

## Interpretation

This is a real benchmark improvement at distances 5 and 7, but not yet the declared
cross-distance win. The failure is localized: distance 3 has only 24 detectors, and
the aggregated morphology head has 11.7% regime error despite temporal filtering.
It throws away which particular detectors and local pairs fired. At distance 5 the
same state reduces mode error from 12.9% to 3.1%; at distance 7 it reaches 0.67%.

The next architecture should remain bounded and causal but expose individual local
detector and neighboring-pair features. The compiler should select, using independent
validation LER, between direct posterior routing and a residual action head trained
only on the rare shots where the two trusted endpoint graphs disagree. This changes
the objective from nuisance classification to value of graph selection without
learning a replacement decoder.

Protocol: [surface-regime-switch-protocol.md](surface-regime-switch-protocol.md).
Artifact: `results/surface_regime_switch.json`.
