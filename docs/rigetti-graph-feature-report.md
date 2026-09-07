# Circuit-local graph-feature screen

The independent soft-parity replay failed, so this screen tests whether cheap
circuit-derived interactions can close the gap to the released stability-9 MWPM
result. Detector coordinates define 578 pairs within one time step and spatial
Manhattan distance 2.01; no logical labels select the pairs. A second feature family
tracks cumulative parity along each of the four detector world-lines.

On the same 60,000/40,000 chronological split, the 109-parameter linear detector
model has 42.395% logical error. World-line parity produces 42.540%, local pairs
42.923%, and their combination 43.118%. None clears the preregistered requirement to
recover 20% of the gap to the released 38.819% MWPM result. The local-pair model's
small Brier improvement (0.24053 to 0.24017) has a descriptive block interval spanning
zero, while its threshold error is worse.

This is a **NO-GO for static feature expansion**. Circuit locality by itself is not
matching: an error-chain decoder must enforce global consistency between syndrome
endpoints and boundaries. The next cheap dynamic test treats each round's four-bit
syndrome as a categorical observation and performs causal class-conditional Markov
updates. Failure there ends hand-built learned decoding and moves the campaign to a
real matching/belief-propagation reference.

Machine-readable results are in
[`results/rigetti_graph_decoder.json`](../results/rigetti_graph_decoder.json).
