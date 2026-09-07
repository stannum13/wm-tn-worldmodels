# Frozen typed-matching transfer

## Question and locked design

The stability-9 exploration selected three effective decoder weights: measurement
`0.001`, one-qubit `0.005`, and two-qubit `0.02`. These numbers are not estimates of
physical device error rates; they compensate for a deliberately approximate circuit
noise model. We froze them and transferred them without fitting or selection to the
independent `stability_8_with_resets_raw_data.h5` acquisition.

The primary endpoint was the maximum-round circuit. The GO condition was at least 1%
relative logical-error reduction over uniform `p=0.002`, with the paired 95% interval
strictly above zero. Intervals are descriptive paired *t* intervals over contiguous
1,000-shot HDF5 row-order blocks. The file has a session-level timestamp but no
per-shot timestamps, so row order is not asserted to be wall-clock chronology.

## Result

| Rounds | Uniform error | Frozen typed error | Absolute reduction | Relative reduction | Paired 95% interval |
|---:|---:|---:|---:|---:|---:|
| 4 | 30.215% | 29.207% | +1.008 pp | +3.34% | +0.877 to +1.139 pp |
| 8 | 25.045% | 24.877% | +0.168 pp | +0.67% | +0.041 to +0.295 pp |
| 12 | 21.295% | 21.466% | -0.171 pp | -0.80% | -0.298 to -0.044 pp |
| 16 | 19.962% | 20.216% | -0.254 pp | -1.27% | -0.383 to -0.125 pp |
| 20 | 18.325% | 18.562% | -0.237 pp | -1.29% | -0.365 to -0.109 pp |
| **24 (primary)** | **16.822%** | **16.993%** | **-0.171 pp** | **-1.02%** | **-0.297 to -0.045 pp** |

The frozen branch is a **NO-GO**. Its shallow-circuit gains reverse by 12 rounds and
the locked primary endpoint shows statistically resolved harm. Looking only at the
4-round gain would therefore have produced the wrong architectural conclusion.

## Scientific consequence

Global gate-type weights do not capture the detector-pair geometry needed at longer
depth. The next experiment must first reproduce the released pairwise-correlation
matching graph. Only after that validity control passes will we test a causal slow
estimator that updates a sparse subset of graph weights while PyMatching remains the
hot decoder. This separates graph misspecification from any benefit due to nonlinear
or history-dependent estimation.

The exact run record, all block differences, timings, data checksum, and source commit
are in `results/rigetti_matching_transfer.json`.
