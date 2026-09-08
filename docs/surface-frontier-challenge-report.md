# Stronger benchmark: repeatable accuracy, bounded deployment claim

8 September 2026. Protocol and implementation were committed at `be70e3e` before
the full evaluation. All twenty independent fits completed; no runs were dropped.

## Nominal result

The adaptation advantage survives a stronger conventional decoder and full
refitting. Each static comparator is selected from 72 models: ordinary and
correlated matching, a 25-point circuit grid, pooled primitive probabilities, and
limited-budget continuous fits. All selection labels are separate from fitting and
final evaluation labels.

| Distance | Selected static LER | Compiled LER | Change, pp | 97.5% interval over full refits, pp | Improving refits |
| ---: | ---: | ---: | ---: | --- | ---: |
| 3 | 2.07443% | 2.01103% | -0.06340 | [-0.08847, -0.03833] | 10/10 |
| 5 | 1.63567% | 1.48605% | -0.14961 | [-0.17197, -0.12725] | 10/10 |

These pass the predeclared minimum-effect gate: both upper bounds are below -0.02
pp. The 97.5% intervals give at least 95% family coverage across two distances.
The experimental units are ten complete calibration/fit/selection/test replicates,
not millions of independent-looking shots under a single fitted model.

The matched history ablation is resolved as well: temporal versus memoryless with
the same backend and action-head complexity improves by -0.04105 pp at distance 3
([-0.05540, -0.02669]) and -0.06653 pp at distance 5 ([-0.07731, -0.05575]). The
correlated-only compiler also wins at both distances. This closes two important
gaps in the preceding result: correlation information and energy features alone
do not explain the whole nominal gain.

## The full robustness gate fails

The study contains 26,214,400 held-out evaluation shots across ten conditions and
two distances, plus separate calibration, action fitting and selection data.

| Test | Result | Reason |
| --- | --- | --- |
| Stronger nominal benchmark | GO | Both distances exceed minimum useful effect; 10/10 improvements |
| Matched temporal evidence | GO | Both paired refit intervals exclude zero |
| Correlated-only benchmark | GO | Correlated adaptation beats correlated static selection |
| Transfer | NO-GO | 7/10 cells have resolved gains, but less-separated d3 upper 99.5% bound is +0.05718 pp, above +0.05 |
| Stationary safety | NO-GO | d3/A upper +0.02679 pp and d5/A upper +0.02087 pp exceed +0.02 |
| Independent-mode safety | NO-GO | Frozen temporal inference harms in every refit at both distances |
| One-decode accuracy | GO at d5; NO-GO at d3 | d3 mean gain is only 0.01984 pp, with 8/10 improvements |

The most informative stress result is a noise switch halfway through the circuit.
Compiled-minus-static degradation is +0.10246 pp at d3 and +0.31975 pp at d5;
97.5% descriptive refit intervals are [+0.05416, +0.15077] and
[+0.29307, +0.34643]. These conditions were reported, not removed from the study.
They are outside the assumed model of one regime per completed shot.

For independent modes, temporal versus the matched memoryless ablation worsens by
+0.04379 pp at d3 ([+0.01671, +0.07087]) and +0.09331 pp at d5
([+0.07987, +0.10674]). More confidence in persistent state is not a remedy when
persistence itself is absent. This motivates a separately frozen
[switching-rate inference experiment](surface-rate-adaptation-protocol.md).

## Native frontend and service cost

At d5, every fit selects a threshold controller. Local detector and pair evidence
is folded into fixed coefficients, and one scalar probability determines the graph
before matching. The compiled representation contains 120 detector terms, 224 pair
terms, 4,552 bytes of float64 constants/index tables and eight bytes of mutable state
per stream, excluding the matching decoder and its graph storage.

All 26.2 million evaluated choices and one-decode predictions match the offline
reference. Arithmetic is not bit-identical: the largest folded/full evidence
difference is about 7.8e-7; none changes a graph choice. The native service path also
has zero prediction disagreements across 20 artifacts x 32,768 records x 3 repeats
for each measured method.

On this Apple M2 Max, d5 static p99 is 38.6–45.2 us and one-decode p99 is
34.8–45.3 us. Paired artifact ratios are 0.895–1.168, inside the 1.25 limit. Mean
one-decode service is 13.8–16.8 us. These are warm, batch-one, completed-shot
measurements including native/Python boundaries and matching; they exclude physical
acquisition. Correlated matching internally performs two blossom passes even though
only one graph is decoded.

Thus d5 supports a nominal accuracy improvement at comparable p99 service cost on
this host. It does not establish deterministic latency: an observed service outlier
reaches 4.6 ms, and the mean path cannot sustain the illustrative five-us shot
arrival scenario. The artifact includes queue replay at specified hypothetical
cadences. At d3, nine of ten selected policies require two decoders; the accuracy
and cost result cannot be transferred from its cheaper threshold lane.

## Audit and limits

An independent review checked all 760 distinct sampler seeds, all 280 distinct
dataset hashes, the identical recorded source hashes, configuration, and recomputed
intervals. It confirmed the positive and negative gates and requested stronger
automatic provenance checks in the summarizer; those checks are included.

All 40 bounded two-parameter circuit optimizations exhausted their evaluation
budgets. Their best evaluated candidates were retained as predeclared; the pooled
scalar optimizations converged. The comparator is a specified candidate family,
not a proof of optimal static decoding. Evidence fitting uses labeled nominal noise
regimes, graph templates are known, the circuits are synthetic Z-memory experiments,
and the controller acts after completed shots. DGR, AlphaQubit, qLDPC decoders, FPGA
implementations and live devices were not reproduced in this comparison.

The supported advance is a repeated nominal adaptation result, with a measured
small-state implementation at d5 and explicit failure boundaries. The complete
robustness-and-cost claim remains NO-GO.

Reproduction: `scripts/run_surface_frontier_challenge.py` with the default frozen
configuration; `scripts/summarize_surface_frontier_challenge.py`; then
`scripts/benchmark_surface_frontier.py`. Use `PYTHONPATH=src:.`, the recorded
Stim/PyMatching/NumPy versions and single-thread BLAS settings from the run command.
Raw artifacts: `results/surface_frontier_challenge/`. Aggregate:
`results/surface_frontier_challenge_summary.json`. Service:
`results/surface_frontier_latency.json`.
