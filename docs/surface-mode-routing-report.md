# Persistent surface-code mode routing

Precommitted synthetic circuit-level benchmark, 8 September 2026.

## Outcome

A causal two-state filter can use persistence across completed surface-code shots to
select between two calibrated PyMatching graphs better than a matched memoryless
selector. This is the first graph-operation result in the campaign on a standard
rotated surface-code circuit rather than the custom parity toy.

| distance | static nominal LER | mode-informed LER | temporal LER | temporal vs memoryless (95% CI, pp) | temporal mode error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.08736% | 0.08011% | 0.08278% | [-0.01018, -0.00203] | 3.34% |
| 5 | 0.01373% | 0.01106% | 0.01068% | [-0.00464, -0.00070] | 2.06% |
| 7 | 0.00114% | 0.00114% | 0.00114% | [0, 0] | 0.85% |

Stim 1.16.0 generated rotated-memory-Z circuits with equal distance and rounds.
Nominal reset/measurement flip probability was 0.001, the persistent burst value was
0.01, and after-Clifford depolarization was 0.001. Each distance used 262,144 test
shots arranged as 512 independent streams. PyMatching 2.4.0 supplied the trusted
constraint-preserving decision layer.

## Precommitted decisions

1. **Graph-template relevance: NO-GO as written.** True-mode graph selection improves
   over the static graph at distance 3 by -0.00725 pp, 95% CI
   [-0.01235, -0.00214]. At distance 5 the point improvement is -0.00267 pp but its
   interval [-0.00537, +0.00002] pp crosses zero. The condition required resolution
   at both distances.
2. **Routing utility: NO-GO as written.** The temporal router recovers 63.2% of the
   mode-informed gain at distance 3, below 80%; at distance 5 its point estimate is
   114.3%, because it slightly outperforms true-mode template selection. The condition
   required both distances.
3. **Temporal-state value: GO.** Temporal routing beats matched memoryless routing at
   distances 3 and 5 with paired intervals wholly below zero. Distance 7 has no
   decoder disagreements between the routers.
4. **Nominal-null safety: GO.** Worst upper interval bound is +0.00168 pp, below the
   fixed +0.01 pp cap at every distance.

“Mode-informed” is intentionally not called a maximum-likelihood oracle. It knows the
generator mode but is constrained to choose that mode's MWPM graph. At distance 5,
the temporal router's posterior shrinkage happens to make slightly better graph
choices than hard true-mode selection. This is evidence that generator-mode accuracy
and logical-decision utility are not identical objectives.

## What the filter contributes

The memoryless burst-selection fractions are 0.08%, 1.45%, and 3.82% at distances
3, 5, and 7. Temporal filtering raises them to 1.99%, 3.90%, and 4.31%, close to the
true burst fractions of 4.19%, 4.92%, and 4.92%, while reducing mode-classification
error at every distance. The effect is strongest where one shot contains too few
detectors to identify the nuisance mode reliably.

This supports a narrow local-sufficient-state claim: recurrent state is valuable when
the observation attached to one graph decision is weak but the nuisance process
persists. It does not show that a generic RNN is needed; the winning state is one
scalar posterior updated by a two-state filter.

## Architectural implication

Whole-graph switching is not the final architecture. The two failed primary
conditions and the fact that an always-burst graph slightly beats true-mode selection
at distance 3 show that global templates conflate useful and harmful edge changes.
The next surface-code experiment should localize the burst and activate only the
affected spacetime-edge class or patch. That directly tests bounded `REWEIGHT` or
`ACTIVATE_MODE(region)` overlays.

This benchmark remains synthetic and shot-causal, not cycle-causal: the router sees
the completed shot's syndrome before decoding. It does not model analog I/Q, leakage,
within-shot mode changes, early decoding, hardware service time, or a live device.

Protocol: [surface-mode-routing-protocol.md](surface-mode-routing-protocol.md).
Machine-readable artifact: `results/surface_mode_routing.json`.
