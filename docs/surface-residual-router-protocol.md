# Residual graph-router compiler protocol

Precommitted confirmatory protocol, 8 September 2026. Final test seeds and outcomes
were not inspected when this protocol was committed. The raw/local feature expansion
and residual action formulation follow the preceding aggregate-router NO-GO.

## Architecture

Keep the two trusted PyMatching endpoint graphs and the same incompatible surface-code
regimes. Replace the aggregate observation head with a bounded affine head over:

- every detector bit;
- every same-position, adjacent-round detector-pair product;
- every nearest-neighbor, within-round detector-pair product;
- aggregate total, temporal-pair, spatial-pair, and per-round counts.

An independent balanced calibration set fits the regime evidence head. A disjoint
persistent validation set then compiles one of two policies:

1. posterior threshold routing, with threshold selected from
   {0.1, 0.2, ..., 0.9};
2. a second affine residual head, evaluated only when the endpoint decoders disagree,
   trained to select which endpoint prediction is correct.

The residual head receives the detailed features plus the causal regime log-odds.
It cannot emit an unconstrained correction: its only legal action is selecting one
of two constraint-preserving endpoint predictions.

## Fixed data and comparators

- Distances=rounds {3, 5}; 512 test streams x 512 shots.
- 65,536 independent shots per regime for evidence calibration and the same 25-way
  static-graph search as the prior protocol.
- 256 x 512 disjoint persistent validation shots for policy compilation.
- Independent final seed 20261123; no final-test tuning.
- Comparators: the selected 25-way static graph, the prior aggregate causal router,
  detailed memoryless routing, detailed causal posterior routing, mode-informed
  endpoint selection, and the compiled residual/posterior policy.

## GO conditions

Paired stream-level 95% t intervals are authoritative.

1. **Benchmark win:** the compiled policy beats the selected static graph with upper
   interval bound below zero at both distances.
2. **Distance-3 frontier:** the compiled policy beats the prior aggregate causal
   router with upper interval bound below zero at distance 3 and is no worse than
   +0.02 percentage points at distance 5.
3. **Oracle recovery:** the compiled policy recovers at least 80% of the point gap
   from selected static to mode-informed endpoints at both distances.
4. **Memory value:** detailed causal posterior routing beats detailed memoryless
   routing with upper interval bound below zero at both distances.
5. **Stationary safety:** the compiled policy is no worse than +0.05 pp by upper
   interval bound versus the corresponding endpoint graph under each stationary
   regime and distance.

Passing establishes an improvement over the declared synthetic Stim/PyMatching
benchmark family. It does not constitute general surface-code SOTA, hardware latency,
or prospective-device evidence.
