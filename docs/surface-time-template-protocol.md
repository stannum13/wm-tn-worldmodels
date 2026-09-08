# Prospective time-resolved graph-template experiment

8 September 2026. Freeze this protocol and implementation before final simulation.
This is a new full-refit campaign, not a relabeling of the previous failed midpoint
control. Root seed 2026090806; ten new d5 fits. No test-based parameter changes.

## Question and scope

Can four-mode causal inference choose among AA, AB, BA and BB correlated-matching
graphs better than the same inference restricted to AA/BB, memoryless selection,
and an independently selected static graph? A/B are the previously specified
measurement- and gate-dominated noise regimes. AB/BA change parameters halfway
through circuit TICKs. Ideal operations and available physical error mechanisms
are unchanged: this is time-resolved potential/template adaptation, not insertion
of missing error factors. Every decision uses a **completed** detector record;
causality is across records, not feedback before a within-record suffix arrives.

## Fits, data access, and candidates

For each of four nominal circuits, draw 32,768 evidence-calibration and a separate
32,768 risk-calibration shots. Fit one balanced four-class affine softmax emission
model (L2=0.01) to the existing fixed detector/local-pair/count feature map, with
mode labels only. No logical target enters this fit. Fit a 4×4 empirical action
loss table on the second partition by replaying every action. Also fit a
4×4×4 difficulty table, with detector-count bins 0, 1–3, 4–7, 8+, shrunk by 100
pseudo-observations toward the corresponding global mode/action mean.

These tables are restricted approximations: syndrome-derived evidence and action
loss need not be independent given the mode. Do not call their weighted scores
exact conditional risks. Record their predicted/observed calibration by confidence
and count bins. Compare three fixed action rules: posterior mode selection, global
loss table, and count-conditioned loss table. Endpoint-only mode selection splits
AB/BA belief equally between AA/BB. Loss-table selection just restricts actions.

State variants share the SAME emission fit: memoryless; fixed symmetric four-mode
transition with total switching probability .02; and a joint rate/mode bank with
rates {.005,.02,.1,.5,.75}, uniform initial belief and uniform rate refresh .002.
At each step refresh the rate hypothesis, transition mode (equal probability to
each alternative), then multiply the current emission once. There are 20 persistent
float64 probabilities (160 bytes), excluding constants and four decoder graphs.
The .75 rate is the four-mode independent limit. The four-mode bank is deliberately
misspecified for streams restricted to two modes; do not import old binary safety.

The static family receives the same risk-calibration shots from all four modes:
the parent 72 candidates (25 two-parameter grids and nine pooled DEMs per backend,
plus four bounded continuous fits), plus explicit AB/BA matching graphs for both
backends. Also include a correlated time-resolved four-parameter static fit, with
64 objective evaluations total: two mirrored AB/BA initial points, 32 evaluations
each, with a +0.2 log-rate coordinate simplex, all four rates bounded to
[0.0003, 0.03]. Minimize equal-mode mean logical error on the first 16,384
risk-calibration shots per mode; retain the best evaluated point across both
starts, ties by first evaluation, even when neither optimizer converges.
There are 77 static candidates. All static and
adaptive candidates are selected using a disjoint 256×512 four-mode stream.
Thus a new time-resolved graph does not enter only the adaptive action family.
Budget-limited optimizers are disclosed, never called globally optimal.

Select each action/state family separately on actual logical loss. These are
capacity-symmetric **family comparisons**, not mechanistically matched ablations.
After selecting the rate-bank four-action rule R4, construct matched R2 by changing
only its allowed actions to AA/BB, and matched M4 by removing only its history.
They retain R4's selected rule, tables, emission fit and backend, with no reselection.
The raw R4 is authoritative for all gates below. Select a
combined controller from the chosen rate-bank four-action family and every static
candidate; report any fallback separately, without substituting it for R4 at a gate.
Ties prefer the mode rule, then global table, then count table, and static fallback.
Each selected adaptive policy chooses a single graph before matching; evaluate
exact equivalence to the corresponding predecoded counterfactual predictions.
Correlated matching still uses two internal matching passes. No latency or hardware
advantage is claimed without a separate end-to-end timing experiment.

## Fresh tests and references

Each condition has 256 independent streams of 512 completed shots per fit:

- four-way AA/AB/BA/BB Markov switching, total probability .02 (primary);
- four-way independent modes (.75);
- persistent AA/BB only (.02), reproducing the old nominal family;
- persistent AB/BA only (.02), the old midpoint-failure family;
- AB/BA with unseen cutpoints at one-third and two-thirds of circuit TICKs;
- stationary AA and BB.

All selection is frozen before these 10,485,760 test shots. Record full per-stream
error counts, pairwise discordant outcomes, mode×action confusion and source/data
hashes. Semantic seed roles are disjoint from the previous four campaigns.
Assert identical DEM target support, including correlation separators and detector
metadata, across AA/AB/BA/BB before running. Keep both directions in the held-cut
generator. Record risk-calibration count-bin sample sizes/shrinkage, risk-table
resubstitution diagnostics and selection risk calibration/regret; none tunes the
100-pseudocount constant.
Known-mode matching is a privileged **reference, not a Bayes oracle**. On held
cutpoints report separately the matched-cutpoint graph and the nominal four-action
reference. Best-of-four outcome selection is only an unattainable opportunity bound.

## Quantitative decisions

Primary unit: complete independently calibrated fit, n=10. Confidence intervals
are paired t intervals on the ten LER differences, not on pooled nested streams.

1. Primary graph/temporal GO: R4 beats strongest selected static and matched R2 by
   at least 0.02 pp at the upper interval endpoint, and beats matched M4.
   Static and R2 upper bounds must be <−0.0002 in probability units; M4 upper <0.
   Use 98⅓% intervals for
   these three contrasts (family 95%); at least 8/10 fits improve on each.
   Separately report a family-benchmark gate: R4 beats independently selected
   two-action and memoryless families, using 97.5% intervals (two comparisons).
2. Mechanism GO: within-shot AB/BA improves at least 15% in mean relative LER versus
   matched R2: (mean LER_R2 − mean LER_R4)/mean LER_R2 across the ten fits, and its
   absolute paired-difference 95% interval excludes zero. Report each
   orientation as well; this pooled criterion cannot establish both separately.
3. Safety GO: exclude more than +0.02 pp versus matched AA/BB graph in both stationary
   conditions and versus matched M4 under independent modes.
   Use 98⅓% intervals for the three safety comparisons.
4. Held-cutpoint GO: improve versus selected static and matched R2 at both
   unseen cutpoints, with 98.75% intervals below zero (four comparisons, family 95%).
   Report per-orientation differences descriptively; do not claim universality.
5. At least 100 total paired correctness discordances per primary contrast (sum
   over fresh test streams, with each fit's count reported), all semantic
   equivalence checks zero, complete provenance and ten fits. No optional stopping
   or enlarging sample size after seeing a gate; an underpowered gate is unresolved.

These families are separately labeled, not one joint 95% claim. Failure of transfer
means halfway-specific templates, not a successful general compilation grammar.
Failure of conditional-risk calibration motivates a richer observation-dependent
action law; it does not justify calling the mode posterior itself wrong.
