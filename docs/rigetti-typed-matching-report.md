# Type-calibrated matching exploration

The uniform PyMatching control reproduces the released stability-9 error scale. This
next mechanism test separates measurement/reset, one-qubit-gate, and two-qubit-gate
error probabilities while leaving the circuit graph and decoder unchanged. A coarse
27-cell grid is evaluated on the first 20,000 shots; its five best cells are selected
on shots 20,000–59,999; the final 40,000 shots remain the test partition.

The selected probabilities are 0.001 for measurement/reset, 0.005 for one-qubit
gates, and 0.02 for CZ gates. On test, error falls from 38.7475% to 38.5375%: a 0.210
percentage-point or 0.54% relative reduction. The paired interval across contiguous
blocks is -0.141 to +0.577 percentage points, and the effect misses the preregistered
0.3875-point threshold. Decode throughput is unchanged within timing noise
(45.8 versus 45.4 us/shot in these calls).

This is a **NO-GO on stability-9 alone**. The direction is physically plausible—CZ
errors receive the highest selected rate—but neither the effect size nor uncertainty
supports a claim. Further tuning on the already observed test session is prohibited.
The exact triplet is frozen for transfer to the independent stability-8-with-resets
acquisition. A positive transfer result would justify continuous drift estimation of
these three slow graph parameters; a negative result ends the typed-rate branch.

Machine-readable results are in
[`results/rigetti_typed_matching.json`](../results/rigetti_typed_matching.json).
