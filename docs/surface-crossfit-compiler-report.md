# Cross-fitted graph-router benchmark

Corrected confirmatory result, 8 September 2026. An initial execution was discarded
after independent audit found reused Stim seed namespaces. All numbers below are from
the corrected rerun at code commit `f499d98`.

## Result

The cross-fitted compiler passes all five declared GO conditions on the incompatible
surface-code regime benchmark.

| distance | selected policy | static benchmark LER | compiled LER | paired change (95% CI, pp) | mode-informed recovery |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | energy residual | 2.0966% | 1.9886% | -0.1080 [-0.1350, -0.0809] | 76.7% |
| 5 | detailed posterior, threshold 0.5 | 1.9089% | 1.7422% | -0.1667 [-0.2007, -0.1327] | 87.8% |

The static comparator is not a nominal straw baseline. For each distance it is the
best mean calibration-LER graph from 25 PyMatching graphs spanning five gate-noise
and five reset/measurement-noise settings. Calibration, residual fitting, policy
selection, final test, and stationary controls use disjoint seed namespaces.

## GO audit

1. **Benchmark win: GO.** Both paired intervals versus the selected static benchmark
   are wholly below zero.
2. **Adaptive frontier: GO.** At distance 3, the compiled residual beats the preceding
   aggregate causal router by -0.0454 pp, 95% CI [-0.0688, -0.0220]. At distance 5,
   the compiler rejects the residual and selects the detailed posterior; its interval
   versus the prior router is [-0.0170, -0.0044] pp, so it improves rather than merely
   meeting the +0.02 pp non-inferiority margin.
3. **Opportunity recovery: GO.** Recovery is 76.7% at distance 3 against a 50% gate
   and 87.8% at distance 5 against an 80% gate.
4. **Stationary safety: GO.** Across two regimes and two distances, the largest upper
   interval bound versus the corresponding endpoint graph is +0.0445 pp, below the
   +0.05 pp margin. This is margin-based safety, not absence of harm: both distance-5
   controls have small resolved degradation (+0.0065 and +0.0076 pp).
5. **Selection integrity: GO.** Calibration uses offsets +100/+101, action fitting
   +1010/+1011, policy selection +2010/+2011, final switching test +3010/+3011,
   and stationary controls +4000/+4001. No final labels select a model or policy.

## What was learned

The winning architecture is not one universally larger model. At distance 3, weak
per-shot observability makes a graph-native residual useful: a 67-input affine action
head sees local detector features, the causal regime log-odds, both MWPM solution
weights, and their difference, and acts only when choosing between the two trusted
endpoint predictions. At distance 5, the compiler correctly rejects that extra head
and retains a 353-feature affine evidence model plus one posterior scalar.

The compiled policy also beats the best detailed memoryless threshold selected on
the disjoint policy-selection stream: intervals are [-0.1310, -0.0811] pp at distance
3 and [-0.0905, -0.0613] pp at distance 5. This establishes a win over the selected
detailed-morphology memoryless family. It does not isolate temporal memory at distance
3 because the residual also receives matching-energy features; a memoryless
energy-residual ablation remains required.

At distance 3, the selected energy residual also beats the closest selected detailed
causal posterior (threshold 0.7) with interval [-0.0546, -0.0110] pp. At distance 5,
the compiler selects the detailed causal posterior itself. There, thresholds 0.5 and
0.7 tie exactly on selection LER (1.7868%); insertion order selects 0.5, so no unique
optimum is claimed.

This is evidence for the architectural thesis:

> Learn the smallest stable adaptation law around a trusted decoder, and select its
> complexity using downstream logical error on disjoint data.

The preceding non-cross-fitted compiler selected the residual at both distances and
regressed badly at distance 5. Adding a separate policy-selection stream changes the
final choice and restores the simpler router. That negative-to-positive transition is
scientifically important: architectural search needs an explicit selection split,
even when every candidate is tiny.

## Claim boundary

This result beats the predeclared 25-candidate, calibration-selected static
PyMatching baseline on this synthetic benchmark, not the general surface-code state
of the art. The physical regimes are deliberately constructed and
known during calibration; the router operates after a completed shot; both endpoint
graphs are evaluated; compute cost and p99 latency are not yet measured; and there is
no prospective device confirmation. The benchmark demonstrates decision value and
cross-fitted architecture selection, not hardware advantage.

The static search is a coarse circuit-parameter grid; it does not include a pooled
mixture detector-error model, continuous edge-weight optimization, or a compute-matched
learned static decoder. The intervals condition on one complete train/select/test
split and exclude fitting and policy-selection variability. Before deployment
framing, scientific gates include repeated complete splits, off-grid rates,
within-shot changes, and stronger static comparators. Engineering gates then include
two-decoder overhead, shared matching, quantization, and measured p99 latency.

Protocol: [surface-crossfit-compiler-protocol.md](surface-crossfit-compiler-protocol.md).
Artifact: `results/surface_crossfit_compiler.json`.
