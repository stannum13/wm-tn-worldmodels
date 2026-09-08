# Independent audit: stronger benchmark and nuisance-rate follow-up

8 September 2026. The reviewer worked from committed protocols, executable code,
raw records and recomputed summaries. This is a separate agent review, not external
peer review or hardware validation.

## Corrections required before execution

The reviewer requested ten complete refits; a correlated-matching comparator;
stronger static fitting; symmetric history ablations; a matched-backend/head
ablation; exact derived seed manifests; and a minimum useful accuracy effect.
It also found that timing a cheap lane while accepting a different expensive lane's
accuracy would disconnect the cost and accuracy claim. The revised protocol requires
the timed one-decode lane to pass its own accuracy condition.

The final implementation includes a correlated-only compiler, tests the complete
folded-feature/scalar-state path against offline decisions, and keeps midpoint
references distinct from an oracle for the actual mixed circuit.

## Parent campaign verdict

PASS for the nominal accuracy, temporal-evidence and correlated-only claims.
NO-GO for overall robustness and deployment: transfer, stationary, independent-mode
safety and d3 one-decode accuracy fail. Service-cost success does not override
those scientific failures.

The reviewer recomputed the summary and verified 20 unique replicate/distance
artifacts, 760 distinct seeds, 280 distinct dataset hashes, identical source-hash
sets matching the frozen commit, all recorded method LERs and confidence intervals.
All compiled-choice and selected-decode differences are zero. Two archival requests
were implemented: enforce both types of equivalence and validate source/config/
version/manifest provenance automatically, including input artifact hashes.

All bounded circuit-parameter optimizers reached their evaluation cap. This was
allowed in the frozen protocol but limits what “optimized static” can mean.

## Exact teacher verdict

The small-code table construction, correlation/separator semantics, conditional
Bayes update and moment validation are sound for this specified DEM. The teacher is
privileged and exponential. Its larger improvement over the compiled policy mixes
decoder expressivity and state estimation; only the direct exact-causal versus
exact-memoryless comparison isolates a temporal benefit within the teacher class.

## Rate-adaptation verdict

PASS for its four declared d5 gates. The joint eight-state filter matches its
specified transition kernel, fresh selection/evaluation roles are independent, and
all source artifact hashes match. The reviewer verified 240 fresh seeds and 80 fresh
data hashes with no parent seed overlap.

Required limits are retained: finite in-grid transition hypotheses, reused evidence
models, small remaining independent-mode harm relative to memoryless inference,
no repair of the parent stationary/observation-shift gates, decisive midpoint harm,
and no measured eight-state latency. One percentage-point conversion in the review
message was corrected against the JSON: the independent-mode noninferiority interval
is [+0.00002, +0.01401] pp, with a lower probability-unit bound of approximately
2.34e-7.

## Conditional-risk distillation verdict

PASS as a scientifically valid negative result. Conditional soft targets and the
action-restricted teacher are constructed correctly, with separate fitting,
selection and evaluation namespaces. All 90 fresh seeds and 30 dataset hashes are
distinct and the source-model hashes match.

Teacher targets improve the selected small head by 0.01602 pp on average (95%
complete-model interval [0.00447, 0.02758] pp improvement; 8/10 improve), but fail
the predeclared minimum-effect condition. Recovery of the restricted teacher's
remaining opportunity is 11.32%, bootstrap interval [5.04%, 17.11%], versus an 80%
target. This rejects this affine/additive-hinge student family, not distillation
generally. There is no matched final-test affine-versus-spline ablation establishing
a KAN-like architecture advantage. Exact targets are privileged simulator-known
probabilities, and the ten underlying evidence models were reused.
