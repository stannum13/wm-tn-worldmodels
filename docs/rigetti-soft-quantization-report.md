# Soft-probability quantization on the Rigetti replay

Exploratory report, 8 September 2026.

## Question and gate

Does a hardware-shaped probability payload retain the logical-error performance of
the floating affine I/Q path on the two already-accessed Ankaa-2 partitions? The
predeclared exploratory gate was: at 8 bits, the upper endpoint of a paired 95% interval
for the absolute logical-error increase must be at most 0.10 percentage points in both
sessions.

The quantizer first computes the probability that the recorded hard bit is wrong,
clips to the existing soft-probability bounds, rounds to the nearest of `2**b` uniformly
spaced reconstruction levels on [0, 0.5] using NumPy's ties-to-even rule, and clips
again. `b=0` is the floating reference. This is motivated by Hanisch et al.'s soft-data
compression result, but the available method description does not establish that this
grid is their exact implementation.

## Result

| session | float LER | 8-bit LER | change | paired block 95% interval | changed decisions |
|---|---:|---:|---:|---:|---:|
| 23 rounds, resets | 16.0625% | 16.0750% | +0.0125 points | [-0.0117, +0.0367] points | 19 / 40,000 |
| 25 rounds, no resets | 18.2775% | 18.2800% | +0.0025 points | [-0.0231, +0.0281] points | 43 / 40,000 |

On these previously accessed Ankaa-2 partitions, 8-bit uniformly quantized soft-flip
probabilities met the exploratory <=0.10-percentage-point noninferiority screen relative
to this pipeline's floating reference. The predictions are not identical.

Seven bits also passed the numerical screen in both sessions, but it was selected from
a multi-bit sweep and is not a confirmatory minimum-precision result. Six bits was
jointly unresolved: the no-reset change was +0.0500 points with an upper endpoint of
+0.1246 points. Four bits caused resolved harm without resets: +0.6700 points,
95% interval [+0.5354, +0.8046]. This sharp asymmetric failure makes the no-reset
partition the useful precision stress test.

## Interpretation and limits

The result supports a one-byte *probability payload* as the conservative native
compiler target and qualitatively extends the direction reported by Hanisch et al. to
a different device and decoding pipeline. Relative to a float64 probability value,
that field alone is eight times smaller. It does not include the hard bit, framing,
indexing, graph storage, or raw-I/Q/control-card processing.

No serialization, bandwidth, FPGA area, power, or latency was measured, so this is not
a hardware-efficiency or deployment claim. The intervals are descriptive across 40
contiguous 1,000-record HDF5-row blocks; acquisition-level independence and chronology
are unavailable. Both sessions had already informed earlier design, so the screen is
exploratory rather than prospective confirmation.

Artifacts:

- `results/rigetti_soft_quantization_with_resets.json`
- `results/rigetti_soft_quantization_no_resets.json`

Code: `scripts/run_rigetti_soft_quantization.py` and
`src/ptwm/rigetti_jit.py::fused_affine_quantized_weights_one`.
