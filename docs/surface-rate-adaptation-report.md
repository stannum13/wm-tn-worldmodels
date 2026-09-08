# A 64-byte nuisance filter repairs most independent-mode harm

8 September 2026. This follow-up was designed after the parent campaign revealed
transition mismatch, then committed and evaluated on fresh selection/test streams.
It reuses ten previously fitted distance-5 evidence models and static comparators.

The additional state is a probability distribution over four switching-rate
hypotheses and two physical regimes: eight float64 probabilities, or 64 bytes per
stream. Current detector evidence updates the joint state; only one selected
correlated matching graph needs to decode each shot. This implementation's latency
has not been measured and cannot inherit the earlier scalar filter's result.

| Condition | Fixed 0.02-switch filter | Rate-adaptive filter | Memoryless | Selected static |
| --- | ---: | ---: | ---: | ---: |
| Nominal persistence | 1.47888% | 1.48155% | 1.54411% | 1.62117% |
| Independent modes | 1.62575% | 1.53320% | 1.52618% | 1.59225% |

All four declared conditions pass:

- Nominal retention: adaptive-minus-fixed is +0.00267 pp, 97.5% replicate interval
  [-0.00015, +0.00549], within the +0.02-pp margin.
- Nominal static win: -0.13962 pp, interval [-0.16194, -0.11730].
- Independent-mode repair: adaptive-minus-fixed is -0.09254 pp, interval
  [-0.10507, -0.08002]. All ten replicate point estimates improve.
- Independent-mode margin: adaptive-minus-memoryless is +0.00702 pp, interval
  [+0.00002, +0.01401], below +0.02. This remains a small resolved harm, not equality
  with memoryless inference; eight replicates worsen, one ties, and one improves.

The filter's posterior mean rate in the final half of each stream is about 0.024
under nominal switching and 0.489 under independent modes. Under a true 0.005 rate
it stays near 0.0115, so this is not an unbiased parameter estimator. All tested
rates are present in the finite hypothesis grid. Off-grid transition learning has
not been established.

Mid-circuit changes still fail: adaptive LER is 1.74080% versus static 1.46393%, a
0.27687-pp degradation ([+0.24482, +0.30893]). There the rate posterior approaches
rapid switching even though the orientation changes slowly between shots. The
controller is using its available rate hypotheses to absorb a different modeling
error: a completed shot no longer has one homogeneous regime.

The study uses 9,175,040 fresh evaluation shots and 1,310,720 fresh nominal selection
shots. Independent review checked all 240 unique new seeds, 80 unique data hashes,
zero seed overlap with the parent campaign, and all referenced source artifact
hashes. Exhaustive small-state, single-rate equivalence, causality and boundedness
tests pass. Statistical intervals condition on the reused evidence-model ensemble;
this is not ten new end-to-end evidence fits.

This establishes a prospective finite-grid repair for one identified failure at
d5. It does not repair the parent campaign's less-separated-noise or stationary
endpoint gates: the former was not rerun, and the latter's endpoint comparator was
not retained in this follow-up. The midpoint failure remains an explicit priority
for graph templates with finer temporal scope.

Protocol: [surface-rate-adaptation-protocol.md](surface-rate-adaptation-protocol.md).
Artifact: `results/surface_rate_adaptation.json`.
