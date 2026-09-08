# Nonlinear particle-filter feasibility screen

This diagnostic asks whether particles justify their cost when the latent detuning is
continuous, the dynamics and observation are nonlinear, and 2% of detector samples
are artifacts. Dynamics and noise parameters are known to every estimator, so this is
a privileged capacity comparison rather than learned deployment.

Ten independent seeds each contain 30 streams of length 800. Predictions target the
latent state ten samples ahead. We compare a bounded-innovation EKF with Gaussian and
contamination-robust bootstrap particle filters. Systematic resampling occurs below
50% effective sample size.

| Estimator | Particles | MSE | Mean time/sample on GCP CPU | Gain over robust EKF |
|---|---:|---:|---:|---:|
| Robust EKF | – | 0.2491 | 19.0 μs | – |
| Gaussian PF | 32 | 0.2466 | 67.2 μs | 1.0% |
| Robust PF | 16 | 0.2419 | 62.9 μs | 2.9% (95% paired interval 2.4–3.4%) |
| Robust PF | 32 | 0.2378 | 67.1 μs | 4.5% (4.1–5.0%) |
| Robust PF | 64 | 0.2360 | 74.4 μs | 5.2% (5.0–5.5%) |
| Robust PF | 128 | 0.2354 | 91.0 μs | 5.5% (5.1–5.8%) |

The robust PF improves monotonically but saturates well below the program's 20%
minimum worthwhile effect while costing 3.3–4.8 times the EKF runtime. It is a
**NO-GO for the hot path in this regime**. Thirty-two particles capture most of the
available benefit and are retained as a slow reference/teacher.

The next particle experiment must use a genuinely multimodal wrapped-phase likelihood,
where a Gaussian posterior is structurally wrong, with a one-dimensional grid filter
as the reference. If particles still fail to close at least 20% of the EKF-to-grid gap,
the branch stops before a quantum trajectory implementation. Complete records are in
[`results/nonlinear_filter_benchmark.json`](../results/nonlinear_filter_benchmark.json).
