# Hard-syndrome pairwise-correlation graph

## Purpose

Caune et al. report that their most accurate offline decoder combines a graph inferred
from pairwise detector correlations with per-shot analog I/Q information. Their graph
and implementation are not released. This experiment isolates the first ingredient:
does a transparent implementation of the Spitz pairwise inversion improve an
access-matched hard-syndrome decoder on a locked row-order holdout?

For each circuit, the first 60,000 HDF5 rows estimate edge probabilities without
logical labels. The last 40,000 rows are decoded once. Candidate topology and logical
fault IDs come from the embedded Stim circuit with the gate rates disclosed in Caune
Supplementary Note 2: measurement and two-qubit probability 0.03 and one-qubit
probability 0.003. The current template omits idle-location faults, so this is not an
exact reproduction of the authors' unreleased graph. Invalid estimates use a declared
`1e-4` probability floor; only one edge reaches it in the larger circuits.

## Results

| Decoding rounds | Circuit template | Pairwise graph | Relative reduction | Paired 95% interval (absolute) |
|---:|---:|---:|---:|---:|
| 3 | 32.050% | 29.323% | +8.51% | +2.328 to +3.127 pp |
| 7 | 25.565% | 24.568% | +3.90% | +0.619 to +1.376 pp |
| 11 | 21.683% | 21.490% | +0.89% | -0.154 to +0.539 pp |
| 15 | 20.275% | 20.203% | +0.36% | -0.225 to +0.370 pp |
| 19 | 18.803% | 18.368% | +2.31% | +0.144 to +0.726 pp |
| **23 (primary)** | **16.515%** | **16.678%** | **-0.98%** | **-0.539 to +0.214 pp** |

The maximum-depth research gate fails. Pairwise graph calibration is beneficial in
several secondary circuits, but a single graph fitted on an earlier row range does not
improve the locked 23-round endpoint. The published soft-plus-pairwise result is
15.901% at this endpoint; comparing it directly to this hard-only split would confound
analog access, graph implementation, and evaluation protocol.

## Directional consequence

The next controlled step is analog information on the same fixed graph, with
cross-fitted per-qubit calibration. Only then is residual temporal variation tested.
If adaptation is needed, a label-free parity-moment ridge/EKF updates graph weights
between completed blocks while PyMatching remains unchanged in the hot path. A
switching filter or particle approximation is eligible only if an EKF fails a locked
next-block calibration test and the inferred state is demonstrably multimodal.

Machine-readable results and graph-fit diagnostics are in
`results/rigetti_pairwise_matching.json`.
