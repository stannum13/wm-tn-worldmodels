# Switching schedule phase diagram — development report — 2026-09-14

## Outcome

The prespecified architectural gate **did not pass**. At nominal noise scale and
64-record dwell, the 256-byte controller reduced aggregate LER from 1.8026% to
1.6337%, while the supplied physical-schedule reference reached 1.3468%. This is
37.1% oracle-gap recovery, not the required 70%.

The improvement was positive in all three independently selected/evaluated
replicates. Per-replicate recoveries were 51.3%, 32.1%, and 28.3%; their
unadjusted three-replicate Student-t descriptive interval is 6.6%--67.9%. Paired
outcomes contain 349 rescues and 266 harms, a net 83 fewer logical errors over
49,152 records. All routed outputs agree with their predecoded graph actions.

The other gate conditions pass. On separate stationary streams the two-change
controller improves, rather than harms, both endpoints: 0.7813% to 0.7060% for A
and 3.0619% to 2.2563% for B. The maximum stationary harm is therefore -0.0753
percentage points. These are development results, not confirmatory estimates.

## Phase diagram

Pooled physical-oracle gap recovery for the 256-byte controller is:

| Dwell (records) | 0.75x rates | Nominal rates | 1.25x rates |
| ---: | ---: | ---: | ---: |
| 4 | 25.4% | -5.4% | -11.3% |
| 8 | 0.0% | -6.3% | -7.5% |
| 16 | 28.8% | -2.6% | -15.2% |
| 32 | -7.3% | 25.7% | 9.9% |
| 64 | 55.1% | **37.1%** | 28.5% |
| 128 | 61.2% | 59.8% | 25.9% |
| 256 | 63.1% | 60.2% | 39.4% |
| 512 | **82.5%** | 42.2% | 56.3% |

The 0.75x/512 cell passes 70%, but it was not the gate and cannot rescue the
nominal/64 claim. Non-monotonic entries are retained. With a fixed 1,024-record
horizon, long-dwell cells contain few transitions and different realized schedule
mixtures; three replicates are insufficient to estimate a smooth boundary.

## What failed

The dominant failure is evidence latency, not absence of a useful action bank.
At nominal dwell 64, the controller chooses the exact graph action on only 33.3%
of records. Only 63.2% of switches settle to two consecutive correct actions
before the next switch, with a censored mean delay of 37.5 records. The explicit
reset fires after 44.0% of switches with a censored mean delay of 45.3 records.
Thus a large part of each 64-record regime is spent carrying stale or ambiguous
state.

The reset heuristic is not a hidden success. On selection data, reset margin 5
beats each replicate's best no-reset configuration by only 2, 0, and 1 errors.
It does not materially solve the tracking problem. The next design needs a direct
sequential likelihood-ratio/change-point state, rather than using disagreement
with a noisy single-record argmax as a reset trigger.

The reachable correct-schedule graph and the privileged physical graph have
almost identical nominal LER, by construction. They also remain close at 0.75x
and 1.25x. Most of the observed deficit is therefore inference/tracking loss,
not merely an unreachable candidate action under rate mismatch.

## State budget is conditional on timescale

At nominal dwell 64, one-change and two-change recovery are nearly identical:
36.2% versus 37.1%, despite 96 versus 256 float64 bytes. At dwell 128 both give
59.8%. At dwell 256, however, two-change reaches 60.2% while one-change reaches
37.7%. A larger grammar only pays once enough observations accumulate to resolve
its extra hypotheses. At short dwell it can be harmful.

This refines the architectural thesis:

> State capacity should be scheduled against environmental persistence. A larger
> graph-intervention grammar without commensurate evidence time increases
> ambiguity faster than it increases useful expressivity.

It also explains why the fixed-condition experiment was much stronger. The
earlier 82.9% recovery measured asymptotic adaptation over persistent streams;
this campaign measures adaptation under repeated state invalidation.

## Scientific implication

The result rejects the simple proposition that a persistent score bank plus a
threshold reset is already an online adaptive decoder. It supports a narrower,
more useful decomposition:

1. **Representation:** the correct graph actions are often available and useful.
2. **Inference:** stationary marginal likelihoods eventually distinguish many
   actions.
3. **Tracking:** current evidence arrives too slowly for 64-record switching.
4. **Resource allocation:** hypothesis width must depend on detected dwell time
   or posterior concentration.

The next intervention should target tracking rather than add neural capacity:
maintain a small run-length/change-point posterior or two-timescale fast/slow
score pair, route initially through the robust one-change bank, and unlock the
two-change bank only when accumulated evidence justifies it. The decisive metric
remains logical regret during the first 16--64 records after a boundary.

## Scope and reproducibility

The campaign uses 1,179,648 fresh evaluation records, plus disjoint calibration,
selection, and stationary checks. It covers eight dwell times, three physical
rate scales, three state budgets, and three replicates. The controller is selected
only on nominal streams at dwell 16/64/256 and is frozen across the phase diagram.

The current record contributes to the graph used to decode that same completed
record. This is not within-cycle feedback, independent hardware data, FPGA
latency, or an external SOTA comparison. Float64 state bytes are an accounting
proxy, not a synthesized implementation.

Frozen protocol: `docs/switching-schedule-phase-diagram-protocol.md`.
Artifact: `results/switching_schedule_phase_diagram_20260914.json`.
Runner: `scripts/run_switching_schedule_phase_diagram.py`.
