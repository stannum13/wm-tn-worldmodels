# Switching schedule phase diagram — frozen protocol — 2026-09-14

This development experiment tests whether the positive persistent-schedule result
survives when the latent within-circuit noise schedule changes between records.
It is frozen before outcome inspection. The target is not used for tuning.

## Physical process and action bank

Use the distance-5 rotated-memory circuit and the two previously defined nominal
noise modes. A record is drawn from one of six representable within-record
schedules: stationary A, stationary B, AB half, BA half, ABA thirds, or BAB
thirds. Each stream holds one schedule for a fixed dwell time and then chooses a
different schedule uniformly. Stream paths and samples use deterministic semantic
seeds and are reset between independent streams.

Compile the 32 nominal graph candidates with at most two changes over six detector
time bins. Compare three state budgets: zero-change (2 candidates, 16 float64
bytes), one-change (12 candidates, 96 bytes), and two-change (32 candidates, 256
bytes). A selected static nominal graph, the nominal graph indexed by the true
schedule (reachable action oracle), and a graph compiled from the true scaled
physical circuit (privileged physical oracle) are controls.

## Controller and selection

For candidate log scores `s[t]`, use

    state[t] = retention * state[t-1] + s[t]

with optional reset. Before the update, compare the best single-record penalized
candidate with the current persistent choice. Reset state when they disagree and
the single-record log-score advantage reaches `reset_margin`. Infinite margin is
the no-reset control.

Select retention from {0.5, 0.9, 0.99, 1.0}, reset margin from {2, 5, 10, 20,
infinity}, and change penalty from {0, 0.5, 1, 2, 4, 8}. Selection uses disjoint
nominal streams at dwell times {16, 64, 256}, pooled within each replicate and
state budget. Freeze one configuration per state budget before evaluation; never
retune by evaluation dwell time or mismatch.

## Phase diagram

Evaluate dwell times {4, 8, 16, 32, 64, 128, 256, 512} and physical noise-rate
scales {0.75, 1.0, 1.25}. Calibration emissions and all reachable graph actions
remain nominal at scale 1.0. This deliberately combines observation and action
mismatch; the two oracle controls separate inference loss from unreachable graph
mismatch.

Use root seed 2026091401 and three replicates. Per replicate, use 65,536 fresh
stationary calibration shots per mode; 8 selection streams of length 512 per
selection dwell; and 16 evaluation streams of length 1,024 per dwell/mismatch
cell. The six schedule identities are balanced at stream starts only; subsequent
transitions are uniform among the other five.

## Outcomes

For every cell and state budget report LER, paired rescues/harms, physical-oracle
gap recovery, reachable-action gap recovery, exact action rate, and errors by
stream. At every true boundary report action settling delay, defined as the first
two consecutive correct schedule actions before the next boundary, censored at
the dwell time. Separately report first reset delay/recall and retriggered resets
per 1,000 non-boundary records. Report excess errors during the first
`min(16,dwell)` records after a switch and the remaining steady portion when it
exists. Retain parameter choices, paths, hashes, source versions, and action-route
equivalence.

## GO condition and interpretation

The prespecified architectural GO condition is at least 70% physical-oracle gap
recovery at nominal scale and dwell 64 for the 256-byte controller, stationary
harm below 0.05 percentage points, positive improvement in every replicate, and
no action-route discrepancies. Mismatch and smaller-budget cells define the
boundary rather than gate the result.

This remains a completed-record simulation: the current record contributes to
the graph selected for that record. It is not within-cycle feedback, FPGA timing,
independent hardware data, or an external SOTA comparison. A negative outcome is
retained and bounds the earlier persistent-condition result.
