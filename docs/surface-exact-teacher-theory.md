# What the smallest exact teacher can tell us

8 September 2026. This is an elementary derivation and a diagnostic protocol, not a
claim of priority for Fourier decoding or hidden-Markov filtering.

## A one-scalar adaptation law exists in the two-regime model

Let `J_m(s,l)` be the joint syndrome/logical probability under regime `m`. Let `b`
be the mode-1 probability before observing the current syndrome. The optimal logical
decision minimizes conditional 0–1 loss:

`predict 1 iff (1-b) [J_0(s,1)-J_0(s,0)] + b [J_1(s,1)-J_1(s,0)] > 0`.

For a fixed syndrome this is affine in `b`. There can be at most one switching
threshold. Thus one scalar can be sufficient for the history while the observation
to decision map is still extremely complicated. Small recurrent state does not imply
a small decoder. It also does not imply that a mixture of edge probabilities can
represent the Bayes action.

After acting, update the mode belief exactly once using the current syndrome:

`b_post = b M_1(s) / [(1-b) M_0(s) + b M_1(s)]`, where `M_m(s)=sum_l J_m(s,l)`.

Propagate `b_post` through the regime transition before the next record. Using the
post-syndrome belief directly to mix *joint* `J_m(s,l)` would count the syndrome twice.

## Why the current log-odds recurrence is stable

For symmetric switching probability `a` in `(0,1/2)`, the prediction step maps old
log-odds `x` to

`f(x)=log[a+(1-a) exp(x)] - log[(1-a)+a exp(x)]`.

Its derivative is

`f'(x)=(1-2a) exp(x) / [(a+(1-a)exp(x))((1-a)+a exp(x))]`.

The denominator minus `exp(x)` is `a(1-a)(exp(x)-1)^2 >= 0`, hence
`0 <= f'(x) <= 1-2a`. Adding the current observation log-likelihood ratio does not
change this contraction factor when comparing histories with identical observations.
For the nominal `a=0.02` filter, initial-state differences contract by at most 0.96
per completed shot. This is a conditional stability result for the filter recursion;
it does not guarantee low logical error under a wrong observation model or wrong
temporal granularity. The midpoint-shift challenge tests precisely that distinction.

## Exact small-code distribution

For independent DEM instructions `(a_j,p_j)` over detector/logical bit masks,
the binary characteristic function is

`phi(k)=product_{j: parity(k & a_j)=1}(1-2p_j)`.

Accumulate `q[a_j] += log1p(-2p_j)/2`, and let `Q=sum q`. With an unnormalized
Walsh-Hadamard transform `H`,

`phi = exp(Q-H(q))`, and `P = H(phi)/2^n`.

Every separated `error(p) ... ^ ...` instruction remains a single Bernoulli fault;
XOR its entire detector/logical support. Treating its components as independent
would erase exactly the correlations the teacher should retain. The undecomposed
DEM is canonical; compare aggregated full masks against the decomposed DEM.

The distance-3, three-round circuit has 24 detectors and one logical bit. Each full
float64 table occupies 256 MiB. Two regime tables are practical offline but scale
exponentially and cannot justify a deployment claim. The native transform is a
reference implementation, not a latency optimization for the production decoder.

## Frozen diagnostic

Use the nominal two regimes of the stronger challenge and its replicate-0 distance-3
selected static and compiled policies. Do not select a favorable replicate using
the new teacher. Use fresh root seed 2026090803, with 512 independent streams of 512
shots for nominal switching and another such dataset for independent modes.
The teacher knows the exact DEM and mode transition; this is explicitly privileged
information, used to measure opportunity and not credited as learned performance.

Require exhaustive small-fault tests, exact hidden-sequence filtering tests, full
mask equivalence before/after DEM decomposition, negative probability mass below
1e-9, normalization error below 1e-10, and sampled Fourier round-trip error below
1e-9. Compare 262,144 circuit samples and 262,144 DEM samples per regime against all
single-bit moments and 64 fixed random parity moments, with a six-standard-error
diagnostic bound. Record every discrepancy before any clipping. These checks validate
this circuit/DEM conversion to sampling precision, not arbitrary conversion flags.

Compare exact memoryless, exact causal with frozen 0.02 switching, exact known-mode,
selected static and selected compiled decoding. For independent modes, also run the
exact filter with correct 0.5 switching; it must reduce to the memoryless teacher.
Report paired intervals across independent streams, retaining logical labels solely
for evaluation. Exact stationary memoryless and known-mode Bayes risks can also be
summed directly, without Monte Carlo noise.

Finally count static failures correctable by choosing between the two endpoint
predictions, and failures the exact teacher fixes when both endpoints are wrong.
These opportunity counts overlap and are not a partition of causal mechanisms.
Teacher-only fixes indicate a limit of that candidate pair; they do not prove that
one specific missing physical mechanism caused the failure.
