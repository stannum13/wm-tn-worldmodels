# Directional feature and schedule screens — 2026-09-09

Development screens, not confirmatory experiments and not SOTA claims. This
protocol is written before the new outcomes are examined. Earlier archived test
partitions are not used for fitting or selection. Existing evidence/static models
are frozen anchors, so the feature screen is **not a full pipeline refit**.

## Feature sufficiency

Use the existing distance-3, two-mode circuit family and exact causal teacher.
Keep both endpoint decoders fixed. Cache their outputs only for offline accuracy
evaluation; both matching energies are inputs, so this is not a one-decode timing
claim. Compare outcome and teacher fitting within each dictionary:

1. Existing detector/pair/action features.
2. Existing features plus connected triple/four-detector XOR motifs.
3. Existing features plus interactions with causal mode belief.
4. Both additions.

All dictionaries are capped at 64 selected columns. Selection is by training-only
correlation with the outcome target on endpoint disagreements, shared between the
outcome-trained and teacher-trained versions of each dictionary. This is a cheap
screen, not an exhaustive feature search. Include the uncapped original affine
outcome/teacher heads as anchors. Fit L2 in {0.001, 0.01, 0.1}; select on separate
selection streams by realized logical loss, and evaluate each frozen choice on
development-evaluation streams. Report conditional-teacher expected risk as well
as realized LER, all rescues and harms, per-stream paired differences, feature
indices, model coefficients, input/source hashes, seeds, and optimizer choices.

Initial root seed 2026090911, replicates 0,1,2, 128 streams x 256 records per
partition (fit, selection, development evaluation). The three partitions total
294,912 records; 98,304 are development-evaluation records. Exact teacher targets
are only valid for this declared nominal two-mode process. This pilot does not
establish held-out-geometry transfer, graph-local memory, or statistical success
at the previous 80% recovery gate. Continue to fresh full refits only if direction
and magnitude warrant it; keep the archived failed result unchanged.

## Temporal schedule compiler integrity and opportunity screen

Generalize fixed A/B cut templates to an explicit mode per circuit tick slot,
including final readout. Compile by selecting aligned operations from unchanged
ideal-circuit templates. This is a noise-schedule compiler, not general composition
of independently decoded patches or arbitrary detector error models.

Compare against the existing splice implementation for AB/BA at 1/3,1/2,2/3 and
ABA at 1/3 then 2/3. Require exact circuit equality, unchanged noiseless circuit,
and detector coordinates. At d3 and d5, sample 32,768 fresh shots per schedule
with semantic seed keys rooted at 2026090912. Compare matching with the actual
schedule (a privileged known-schedule reference), the same-orientation midpoint
graph, and both stationary endpoint graphs. These are opportunity/integrity
results, not an inferred-cut policy or proof of the 70% transfer gate. Report both
rescues and harms; no static candidate is selected using evaluation labels.

## Architectural scope

The feature extension changes the observation-to-action function; it does not
change matching. The schedule extension changes which valid noise graph can be
represented; it does not infer the schedule. Neither screen implements sheaf
learning, boundary-message approximation, or a categorical rewrite engine.
Those remain separately testable candidates, not prerequisites for these runs.

## Follow-up: bounded inferred-schedule development screen

This follow-up is frozen before inspecting its outcomes. At distance 5, use only
stationary A/B calibration shots to estimate smoothed Bernoulli emissions for
each detector. Group the six detector-time coordinates into a binary schedule.
Enumerate all schedules with at most zero, one, or two changes (2, 12, and 32
candidate sequences). For a candidate sequence, score each record by the sum of
per-detector stationary log likelihoods minus a selected change penalty. Compile
each sequence into one fixed-ideal-circuit graph, placing detector-bin boundaries
at the corresponding circuit-tick fractions. This approximation does not model
cross-boundary detector dependence.

Use separate stationary calibration, balanced schedule-penalty selection, and
development-evaluation partitions. Selection and evaluation contain stationary
A/B, AB/BA cuts at 1/3, 1/2 and 2/3, ABA/BAB cuts at thirds, and AB/BA cuts at
1/4 and 3/4. The quarter cuts are not exactly representable by six detector bins;
the actual physical schedule graph is retained as a privileged reference. Select
the change penalty separately for the 0/1/2-change families from
{0, 0.5, 1, 2, 4, 8}, then select one fixed compiled graph on the same selection
mixture as a static comparison. Neither selection can use evaluation labels.

Initial root seed 2026090913, three independent replicates, 65,536 calibration
shots per stationary mode, 4,096 selection and 16,384 evaluation shots per
condition. Use common detector records for all decoders within a condition.
Report complete condition-level logical loss, inferred sequence confusion,
rescues and harms relative to the selected static graph, data/seeds/source hashes,
and exact action-pipeline equivalence. This is a development screen, not the old
ten-refit 70% transfer gate and not a SOTA comparison.
