# Four time-resolved graphs repair the midpoint failure, not arbitrary timing

8 September 2026. Ten new complete d5 refits; **10,485,760 fresh held-out shots**.
Protocol and executable sources were frozen at commit `88122c5` before execution.
The [protocol](surface-time-template-protocol.md),
[raw artifacts and summary](../results/surface_time_templates/), and
[runner](../scripts/run_surface_time_templates.py) are public and reproducible.

## Outcome

All ten fits improve the primary four-mode benchmark. A 20-probability causal
controller selects one of four correlated-matching graphs before decoding.
It beats the strongest selected static candidate, matched endpoint-only actions,
and matched memoryless selection. The trained midpoint failure is substantially
repaired. **The held-cutpoint transfer gate fails**, including resolved harm when
the switch moves to two-thirds of the circuit. There is no overall robustness GO.

| Fresh test | Selected static LER | Four-action causal LER | Causal − static, percentage points | 98⅓% complete-refit interval |
| --- | ---: | ---: | ---: | --- |
| Four-way AA/AB/BA/BB | 1.536102% | 1.423721% | −0.112381 | [−0.137874, −0.086888] |
| Persistent AA/BB | 1.628342% | 1.483383% | −0.144958 | [−0.171108, −0.118809] |
| Persistent midpoint AB/BA | 1.453857% | 1.326065% | −0.127792 | [−0.145380, −0.110205] |
| Independent four-way modes | 1.532745% | 1.521454% | −0.011292 | [−0.039785, +0.017202] |
| Unseen one-third cut | 1.518631% | 1.529160% | +0.010529 | [−0.020737, +0.041794] |
| Unseen two-thirds cut | 1.462250% | 1.509552% | +0.047302 | [+0.025535, +0.069069] |

These table intervals use one consistent display level. The frozen four-comparison
transfer family uses 98.75% intervals: [−0.022630, +0.043687] pp at one-third and
[+0.024217, +0.070387] pp at two-thirds. Claims use their specified family, not the
most favorable interval. Primary improvement versus static is 7.32% relative;
midpoint improvement versus static is 8.79% relative.

## What was actually compared

AA and BB use the two stationary noise regimes; AB/BA splice them halfway through
the 35 circuit ticks. Fractional cutpoints are rounded down by the generator:
one-third, halfway and two-thirds mean tick boundaries 11, 17 and 23.
All four flattened detector-error models contain 2,073
aligned instructions with identical target support and correlation separators.
This changes time-resolved potentials, not physical error support or graph topology.

Each fit uses 32,768 mode-evidence shots and an independent 32,768 action-risk
shots per regime. One four-class affine model sees detector bits, predeclared local
pairs and counts, supervised by noise mode only. Global and four-bin action tables
use logical outcomes on the separate risk partition. A 256×512 selection stream
chooses the rule before any final test is used.

The static baseline is selected from **77 candidates** on the same selection data,
with the same four risk-calibration arms: ordinary and correlated matching grids,
pooled DEMs, budgeted circuit/mixture fits, explicit AB/BA graphs for both backends,
and one four-parameter temporal fit using mirrored AB/BA starts. Each temporal fit
uses all 64 allocated evaluations; these are bounded searches, not proofs of the
best possible static weights.

The state bank has five rate hypotheses and four modes, using 160 bytes of
persistent float64 probabilities. The selected four-action rule is global-risk
lookup in 5/10 fits, direct mode selection in 3/10, and count-conditioned lookup in
2/10. The combined compiler selects the adaptive path in all ten fits. Selection
frequencies alone do not establish that one rule family is superior.

## Memory and action access are separate effects

Both matched ablations retain the four-action winner's selected rule, tables,
emission model and correlated backend. One removes history; the other restricts
available actions to AA/BB without selecting a different rule.

| Primary contrast | Reference LER | Four-action − reference | 98⅓% interval, pp | Improving fits |
| --- | ---: | ---: | --- | ---: |
| Strongest selected static | 1.536102% | −0.112381 pp | [−0.137874, −0.086888] | 10/10 |
| Matched AA/BB-only actions | 1.601105% | −0.177383 pp | [−0.204092, −0.150675] | 10/10 |
| Matched memoryless state | 1.544952% | −0.121231 pp | [−0.129255, −0.113207] | 10/10 |

The independently selected two-action and memoryless families reach 1.583633% and
1.542740%, respectively. The separate family-benchmark gate passes as well.
There are 10,971, 7,231 and 4,853 paired correctness discordances in the three
primary contrasts—well above the 100-discordance requirement.

On the midpoint family, matched two-action error is 1.681747% versus R4's 1.326065%:
a **21.15% relative reduction**, satisfying the 15% mechanism gate. Both directions
have favorable point differences versus static (AB −0.144761 pp, BA −0.110701 pp);
orientation figures are descriptive, not extra preregistered significance claims.

## Safety is noninferiority, not zero harm

Stationary A/B differences against their matched known-mode graphs are +0.002594
and +0.003586 pp. Their 98⅓% intervals are [+0.000184, +0.005004] and
[+0.002019, +0.005152] pp: small resolved harms, both inside the +0.02-pp margin.
Independent-mode difference against matched memoryless inference is +0.001144 pp,
interval [−0.002972, +0.005261]. The safety gate therefore passes, but it would be
incorrect to say that adaptation never hurts.

The fixed-rate four-state selector reaches 1.422195% on the primary family versus
1.423721% for the rate bank. This experiment does not establish a nominal benefit
from carrying all five rate hypotheses; the bank is intended to handle unknown
persistence. Its cost cannot be justified from nominal error alone.

## Why the transfer failure matters architecturally

At one-third, R4 is 1.529160%; selecting the halfway graph with the true direction
still gives 1.497803%, while a privileged graph built at the actual cut reaches
1.355057%. At two-thirds those values are 1.509552%, 1.469498% and 1.309509%.
Knowing the direction does not repair a wrongly placed change boundary.
These references are matching decoders, not Bayes optima. In particular they do
not prove that no other member of the original library could perform better.

Nevertheless, the much larger matched-cut reference opportunity points to time
resolution of the graph law, not simply a larger mode classifier. The smallest
next compositional hypothesis is a one-change-point program built from stationary
A/B primitives and mechanism timestamps. It would generate potentials for an
inferred cutpoint rather than separately learning every AB/BA position. That
proposal requires a new prospective test; it has not passed this one.

## Integrity and limits

The recorded 530 sampler seeds and 170 dataset hashes are distinct, with no seed
overlap with the previous four campaigns. Source hashes resolve to the frozen
commit, dependency versions agree, and all LERs reproduce from paired per-stream
counts. Counterfactual versus selected-only decoding has zero differences across
all 10,485,760 test records. The full fits, not the nested streams, are the units
for reported uncertainty.

Mode/risk tables are not exact syndrome-conditioned risks; calibration diagnostics
are included rather than using a Bayes-optimal label. Across fits, aggregate
predicted-minus-observed risk is +0.00983 pp on the primary family, +0.04803 pp at
midpoint, +0.11331 pp under independent modes, −0.06050 pp at one-third and
−0.01977 pp at two-thirds. Aggregate agreement would not establish correct action
rankings in each stratum. Some zero-count risk-calibration bins contain only
53–68 observations and receive 60–65% shrinkage weight, precluding precise fine-bin
calibration claims. Calibration mode labels and
the template construction are supplied by the controlled simulator. This is not
a calibrated physical leakage model, a DGR comparison, general decoder SOTA,
cross-distance confirmation, or online action before the detector suffix arrives.
Native service profiling is a separate follow-up under its
[own protocol](surface-time-template-latency-protocol.md); accuracy alone supplies
no real-time deployment claim.
