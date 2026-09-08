# Rare local surface-code overlay

Precommitted synthetic circuit-level result, 8 September 2026.

## Outcome

The rare left-half reset/readout burst does not create enough logical-decision
opportunity for the local overlay to beat an independently selected fixed graph.
Local detector counts improve nuisance-mode classification, but every performance
condition except nominal-null safety is a NO-GO.

| distance | static nominal LER | mode-local LER | local temporal LER | selected fixed graph | local temporal vs fixed (95% CI, pp) |
| ---: | ---: | ---: | ---: | --- | ---: |
| 3 | 0.05417% | 0.05264% | 0.05341% | global-burst | [-0.00236, +0.00999] |
| 5 | 0.00458% | 0.00381% | 0.00420% | local-burst | [-0.00226, +0.00074] |
| 7 | 0.00038% | 0.00038% | 0.00038% | local-burst | [-0.00113, +0.00037] |

The fixed benchmark was chosen on 65,536 independent shots per physical mode using
the stationary mode weights. No test labels entered that choice.

## Precommitted decisions

- **Local-overlay relevance: NO-GO.** Mode-informed local graph selection has a lower
  point LER at distances 3 and 5, but both paired intervals versus nominal cross zero.
- **Spatial specificity: NO-GO.** Correct local and wrong global burst graphs are not
  distinguishable at either primary distance.
- **Local observability: NO-GO.** Local-count and global-count temporal routers are
  not distinguishable in logical error.
- **Temporal value: NO-GO.** Local temporal versus memoryless intervals cross zero at
  distances 3 and 5; distance 7 has no disagreements.
- **Benchmark win: NO-GO.** The local temporal router does not beat the selected fixed
  graph at either primary distance.
- **Null safety: GO.** The largest upper bound versus nominal is +0.00030 percentage
  points, below the +0.01 pp cap.

At distance 5, local detector evidence reduces mode error from 4.12% with global
counts to 3.35%, but the router changes too few logical decisions for that state
improvement to matter. This is precisely why mode-estimation accuracy cannot serve as
a surrogate endpoint for decoding.

## Consequence

The next benchmark should not make the adaptive mechanism compete against a fixed
graph when one physical mode is rare and one conservative graph works well in both.
A sharper test is a persistent hotspot that alternates between disjoint left and
right regions. There, local graph templates make incompatible weight changes, and
the strongest fixed global or single-region graph remains a meaningful comparator.

This is a benchmark redesign motivated by a resolved lack of logical opportunity,
not parameter tuning on the same test. The rare-burst result remains the negative
control and will not be discarded.

Protocol: [surface-local-overlay-protocol.md](surface-local-overlay-protocol.md).
Artifact: `results/surface_local_overlay.json`.
