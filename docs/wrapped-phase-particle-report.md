# Wrapped-phase particle positive control

## Question and decision rule

The earlier double-well screen did not require a multimodal posterior and produced
only a 5.5% particle-filter gain. This positive control asks the narrower structural
question: when the observation symmetry really makes a Gaussian posterior wrong, can
a small particle filter recover a useful fraction of the attainable benefit?

The synthetic stream is a drifting phase on the circle observed through noisy cosine
readout. Seven measurements use the same probe phase, creating a `+phase/-phase`
ambiguity, and every eighth uses an orthogonal probe. Two percent of observations are
broad artifacts. All estimators receive the true dynamics and noise parameters. The
endpoint is ten-step-ahead circular loss, `mean(1-cos(prediction-target))`.

The preregistered GO condition was mean recovery of at least 20% of the bounded EKF's
loss gap to a 512-bin Bayes grid, with the paired 95% t-interval above zero.

## GCP CPU result

The run used ten independent seeds, 30 streams per seed, and 800 observations per
stream (240,000 samples). The 256- and 512-bin grid filters agree to displayed
precision, providing an empirical grid-convergence check.

| Estimator | Capacity | Circular loss | Gap closed | 95% paired interval | Mean time/sample |
|---|---:|---:|---:|---:|---:|
| Bounded EKF | – | 0.05805 | 0% | – | 12.9 us |
| Bayes grid | 256 bins | 0.04419 | 100% | – | 72.7 us |
| Bayes grid | 512 bins | 0.04419 | 100% | – | 92.5 us |
| Robust PF | 32 particles | 0.04820 | 70.6% | 52.5–88.7% | 43.4 us |
| Robust PF | 64 particles | 0.04530 | 91.3% | 84.7–98.0% | 45.9 us |
| Robust PF | 128 particles | 0.04486 | 94.9% | 91.3–98.6% | 52.0 us |
| Robust PF | 256 particles | 0.04461 | 97.1% | 95.1–99.1% | 64.0 us |

The particle branch is a **GO for a slow multimodal inference lane**. Sixty-four
particles are the knee: they recover 91% of the reference gap, while doubling to 128
recovers only another 3.6 percentage points. It is still a **NO-GO for the hot path**:
64 particles cost 3.5 times the EKF in this Python/CPU implementation, and neither is
near a sub-microsecond QEC deadline.

## What this does and does not establish

This is scientifically useful because it falsifies the broad claim that particles
help merely under nonlinear noise while supporting the narrower claim that they help
when posterior topology is genuinely multimodal. It selects posterior geometry—not
model fashion—as the routing criterion between EKF and particle lanes.

It does not establish quantum-control benefit, learned-model benefit, or deployment
latency. The simulator and filters share known parameters, the phase process has no
measurement backaction, and timing is batch replay on a GCP CPU. The next test must
preserve the same comparison against a numerical reference inside a
backaction-consistent quantum trajectory or a real measurement replay.

Machine-readable seed-level records are in
[`results/wrapped_phase_benchmark.json`](../results/wrapped_phase_benchmark.json).
