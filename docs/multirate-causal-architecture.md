# MRAF: multirate robust actuation forecast

## Architectural claim

The proposed contribution is not “KAN for quantum control.” It is a testable
architecture for corrupted streams and delayed actuation: a bounded-influence causal
filter and analytic actuation-time forecast on the hot path, with an asynchronous
slow calibrator that updates drift, contamination, and transition hazards.

```text
stream -> robust fast belief -> actuation-time forecast -> decision
                    ^
          versioned slow calibrator
          (linear, spline, or other head)
```

The hot loop never waits for the slow head. A completed slow update is consumed at the
next tick. A KAN-inspired additive spline is one calibrator candidate, compared with a
linear head, tiny MLP, and lookup table under matched cadence and cost.

## Bounded-influence filter

For a binary latent condition, first predict the belief

\[
\bar b_t=p_{01}+(1-p_{01}-p_{10})b_{t-1}.
\]

Use a contaminated emission likelihood

\[
g_s(y)=(1-\epsilon)\mathcal N(y;\mu_s,\sigma_s^2)+\epsilon q_{\rm tail}(y)
\]

and clip the observation log-likelihood ratio to \([-c,c]\). One arbitrary detector
sample can then change the log odds by at most \(c\). The actuation-time forecast at
delay \(d\) is \(q_t=b_tP^d\). In the two-state case,

\[
q_t=\pi+\lambda^d(b_t-\pi),\qquad \lambda=1-p_{01}-p_{10}.
\]

This equation also corrects the interpretation of the first streaming screen:
forecasting must be compared with the stationary prior, and its advantage over a
deliberately stale current belief is not an architectural discovery.

## Compute–freshness feasibility certificate

Assume the fast filter is \(\kappa\)-contractive in belief and \(L_\theta\)-Lipschitz
in slow parameters. Let slow-head estimation error be \(\eta\), parameter drift per
sample be \(v\), implementation error be \(\xi\), and refresh interval be \(K\). Then

\[
e_t\leq \kappa^t e_0+
\frac{L_\theta(\eta+vK)+\xi}{1-\kappa}.
\]

If fast and slow costs are \(C_f\) and \(C_s\), samples arrive every \(\Delta\), and
allowed utilization is \(\rho\), stable scheduling requires

\[
K\geq \frac{C_s}{\rho\Delta-C_f}.
\]

For filter-error budget \(E\), freshness requires

\[
K\leq
\frac{((1-\kappa)E-\xi)/L_\theta-\eta}{v}.
\]

An empty interval means no refresh cadence can meet both latency and prediction error.
That is a useful engineering NO-GO: improve implementation, estimation, or the sensor
timescale before searching architectures.

## First causal-head result

On the difficult synthetic low-signal/high-artifact setting, a supervised 31-parameter
additive spline head improves Brier loss over a 6-parameter linear head by only 1.21%
(95% paired t interval 0.88–1.54%) at stride 1 and 0.68% (0.46–0.90%) at stride 8.
At stride 32 the interval crosses zero; at 128 the spline is worse on average. Its
per-update cost is roughly 6–8 times higher. This version is therefore a **NO-GO**
under the program's minimum-effect rule.

The linear head retains 95% of its stride-1 predictive performance at stride 32 while
amortized measured head evaluation falls from about 85 ns to 2.5 ns per sample. These
are vectorized GCP CPU timings, not batch-one p99 or FPGA latency. Targets are hidden
simulator states, so the result is a privileged architecture diagnostic. Full records
are in [`results/causal_head_benchmark.json`](../results/causal_head_benchmark.json).

## Next falsifiable contribution

Implement the slow head as a calibrator of physical emission/hazard parameters rather
than a direct state predictor. Compare Gaussian, clipped-likelihood, contaminated-
emission, linear-calibrated, spline-calibrated, and oracle filters. Retain fit
diagnostics and use fresh parameter draws. Only proceed to interactive quantum control
if robust filtering improves fresh-instance reducible Brier loss by at least 20%
without more than 5% clean-regime degradation.

Original KAN reference: [Liu et al.](https://arxiv.org/abs/2404.19756).

## Particle lane

Particles are justified only for continuous, hybrid, or multimodal uncertainty—not
merely because detector noise is non-Gaussian. The cheapest configuration uses
systematic resampling when effective sample size falls below half the population and a
contaminated likelihood that prevents one artifact from collapsing the weights.

For a quantum trajectory, use a Rao–Blackwellized filter: particles represent unknown
detuning, efficiency, or jump mode, while each particle's conditional density matrix
is propagated by the physical stochastic-master-equation update. The slow calibrator
may update proposal spread and contamination parameters atomically between filter
steps; it must not rewrite live weights asynchronously.

The first nonlinear screen finds only a 5.5% MSE gain at 128 particles for 4.8 times
the robust-EKF runtime. Particles therefore remain outside the hot path pending a
multimodal wrapped-phase positive control. See the
[particle report](nonlinear-particle-filter-report.md).

The subsequent [wrapped-phase positive control](wrapped-phase-particle-report.md)
changes the routing decision but not the latency decision: 64 particles recover 91%
of the EKF-to-grid loss gap when sparse quadrature probes create and then resolve a
genuinely multimodal posterior. They cost 3.5 times the EKF, so particles advance as a
slow teacher/reference or event-triggered lane, not as an always-on hot-path filter.
