# A concrete search space for bounded QEC adaptation

Status: architectural specification, not an implemented sheaf decoder or a SOTA
claim. The concurrently implemented experiments are documented in
`directional-feature-screen-protocol.md`. They add feature dictionaries and a
known-noise-schedule compiler. They do not implement the whole system below.

## 1. What physically changes in the software architecture

Keep the code, syndrome extraction circuit, and conventional decoding backend.
Insert an optional small controller before graph selection/publication:

```text
completed detector record + available reset/analog metadata
                    |
        bounded local feature dictionary
                    |
       state update / optional transport residuals
                    |
        small intervention-risk head
                    |
       select/publish permitted decoder graph
                    |
        conventional decoder -> logical decision
```

There is a separate optional local strong-inference lane. Its routing is a later
experiment, not implicitly included in a pre-decode speed claim. The current
feature screen uses energies from **two** decoded endpoints and is therefore an
offline action-selection diagnostic, not the one-decode pipeline above. A first
deployment candidate must remove those inputs or explicitly budget both solves.

Three replaceable implementation blocks, with distinct hypotheses:

| Block | Existing behavior | Candidate change | Scientific question |
| --- | --- | --- | --- |
| Feature/state | Detector bits, pair products, global nuisance belief | Sparse higher-order parities, belief interactions, then local two-channel state | What information is missing from the decision function? |
| Relation operator | Fixed hand-chosen locality/counts | Calibrated typed comparisons or 0–2 sheaf diffusion passes | Do cross-region consistency relations improve decisions beyond equal-cost message passing? |
| Graph publication | Small fixed time-template catalogue | Compose fault probabilities from an inferred schedule on one fixed ideal circuit | Is the transfer failure caused by inadequate graph representation? |

The first screen changes the action function, the second changes the graph
representation. Holding one fixed while testing the other is essential.

## 2. A concrete sheaf candidate, rather than a general sheaf network

### Nodes and state

Use physical stabilizer sites/local neighborhoods as nodes, not a fresh trainable
embedding for every time-expanded detector. Start with two channels per site:
gate-fault evidence and readout/reset-fault evidence. Channels are calibrated
proxies, not proven sufficient statistics or identified physical probabilities.

Construct their inputs from counts, repeated detections, and circuit-derived
parity motifs in the completed record. A future within-cycle version can only
use the prefix available at its declared commit time. Full-record temporal
contrasts cannot be used before the end of the record.

For a distance-5 patch with 24 stabilizers, two int16 state channels would occupy
96 bytes; double buffering makes that 192 bytes. This is only state accounting,
not total memory: coefficients, graph tables, detector buffers, feature indices,
scales, scratch, and controller state must also be reported.

### Relations and restrictions

Edges mean two local views predict something about the **same specified fault
class**, based on circuit support. They do not mean neighboring noise estimates
must be equal. Let each incidence store a 1x2 restriction row:

    r_e = R[u,e] @ z[u] - R[v,e] @ z[v]

Start with fixed calibrated rows, tied by circuit-incidence type. Candidate rows
may later be learned offline, normalized, and quantized into a small dyadic
alphabet. An arbitrary learned row loses its claimed physical interpretation;
provenance and calibration ablations must distinguish these cases.

Two arms are separate:

1. **Residual telemetry:** expose bounded residual magnitude/run statistics to
   the risk head without changing z through diffusion.
2. **Reconciliation:** one or two local passes using delta and its transpose.

No dense Laplacian is formed on the hot path. Each pass consists of gather,
multiply-add, and scatter operations over fixed sparse incidences. With fixed
maps, L = delta.T @ delta. Use

    z_next = clip((1 - epsilon) * (I - eta * L) @ z + B @ local_features)

and a certified upper spectral bound Lambda with 0 <= eta <= 2/Lambda. The
ideal real-arithmetic recurrence is contractive in prior state for fixed inputs
and maps. Quantization/roundoff require their own error and overflow analysis.
State-dependent maps or gates require re-deriving the stability bound, not
reusing the fixed-map argument.

### What would count as a sheaf-specific benefit?

Compare against independent local EMA, ordinary diffusion, and an equal-budget
typed linear message-passing layer. Also shuffle restriction-map assignments.
If a generic layer matches performance, the representation may still be useful,
but a sheaf-specific architecture claim has not been demonstrated.

Simple binary residual squares are already in our feature span:
(s_u-s_v)^2 = s_u+s_v-2*s_u*s_v. Fixed linear diffusion followed by a sufficiently
expressive linear readout also need not add expressive power. Search must remove
these duplicates before attributing gains to topology.

Prior inspiration: [PolyNSD](https://arxiv.org/html/2512.00242v1) studies cheap
restriction maps and bounded-hop filtering. Its reported benchmarks do not
establish effectiveness in temporal QEC; the above is a proposed application.

## 3. The symbolic program and what the compiler checks

Use a small typed dataflow IR, implemented initially as ordinary Python
dataclasses/arrays. There is no requirement to install a category-theory library.

```text
SignalType = (shape, site_support, available_time, evidence_sources,
              logical_frame, scale, numeric_bounds)

Primitive = (input_types, output_types, state_bytes, scratch_bytes,
             coefficient_bytes, operation_count, semantic_kind)
```

| Primitive family | Initial choices |
| --- | --- |
| Observation | COUNT, PAIR, XOR3, XOR4, prefix-available TEMPORAL_CONTRAST |
| State | NONE, EMA, bounded change-detector; at most 2 channels/site initially |
| Relation | NONE, ordinary diffusion, typed residual, bounded sheaf diffusion |
| Reduction | fixed local sum/max; bounded top-k only if needed |
| Decision | affine or shallow spline risk head; bounded threshold/LUT |
| Action | keep base graph, choose existing template, later publish valid schedule weights |

Every proposed program is rejected if it accesses future observations, mixes
logical frames without a map, violates state/overflow limits, uses an uncertified
feedback recurrence under a contraction requirement, or lacks a costed action
implementation. Likelihood multiplication requires declared conditional
independence; reusing evidence in multiple features is allowed but does not make
those features independent observations. Marginal probability, log odds, matching
energy, and empirical confidence are different types, not interchangeable scalars.

Category-theoretic correspondences are **compiler laws**, not additional layers:

- Composition: compatible boundaries and availability times must match.
- Stochastic composition: preserve joint variables instead of inventing
  independence at module boundaries.
- Exact rewriting: simplify a subprogram only when its typed behavior is
  preserved (including temporal state and numerical qualifications).
- Approximate rewriting: quantization, pruning, or finite-state replacement
  needs a declared loss budget and replay evaluation; it is not an equality.
- Resource composition: reject infeasible pipelines. Component p99 values are
  not simply additive; use full service traces or justified tail bounds.

## 4. How neurosymbolic architecture search would actually run

### Outer loop: search discrete programs

First enumerate a small fixed grammar, not arbitrary networks. A first relation
screen can be 4 relation operators x 3 memory choices x 2 widths = 24 structures,
with a fixed action library. Hops, quantization and routing are subsequent stages
only if this screen shows an advantage. This prevents a giant simultaneous
feature/memory/action search from obscuring where gains arise.

Mutations can ADD/REMOVE a motif, change a memory time constant, substitute a
relation operator, share coefficients by incidence type, or quantize a fixed
operator. Do not introduce a new correction action when testing whether a better
feature representation explains the gain.

### Inner loop: fit continuous parameters

Fit risk-head coefficients and, for learned-relation arms, restriction maps on
fitting data. Small differentiable layers can be trained with the existing Torch
dependency; fixed feature programs can keep using SciPy. A rich teacher supplies
conditional action risks during development; logical labels from simulated or
known-state memory experiments provide the outcome-trained baseline.

Train decision quality, not consistency energy alone. A zero restriction map
trivially makes consistency residuals vanish while conveying no useful
information. Norm constraints, fixed-map arms, and downstream evaluation are
mandatory. The same warning applies to teachers trained to imitate other latents.

### Selection and archive

Fit and rank on separate development partitions. Archive nondominated programs
on logical loss, measured end-to-end tail latency, memory and escalation; do not
collapse all dimensions into an arbitrary reward sum. Invalid programs cannot
enter the archive even if a surrogate predicts good accuracy.

Use common trajectories for paired accuracy comparisons; count all harmful
overrides. Cached candidate outputs make screening cheap but are not deployment
timing measurements. A learned surrogate may prioritize trials, never certify
archive admission. Exhaustive enumeration or regularized evolution is sufficient
initially; RL is justified only if it beats those search baselines at equal total
evaluation/training budget.

### Finalists

Freeze architecture and hyperparameters. Refit from fresh calibration data and
evaluate new streams, unseen locations/timing, both logical bases, and larger
distances. Fit the final low-precision program and test it against the reference
implementation before measuring batch-one latency. Retain failed/null regimes.
Repeated examination of one development set is search, not an expanding holdout.

## 5. A higher-risk extension: logical-sensitivity interfaces

If feature search fails, do not just enlarge the feature alphabet. Compare a
different intermediate object: local boundary tables M_R(b, ell; observed local
syndrome), where ell is the region's contribution to a fixed global logical row.
Keep boundary variables and cross-region faults explicit, sum out interiors, then
join compatible tables. Exact tables are an offline small-code reference and
scale exponentially in separator size.

Test compression of these tables or intervention-risk differences, not only
noise-state compression. A new
[August 2026 paper](https://arxiv.org/html/2609.00169v1) already trains noise
parameters on logical outcomes, studies exact distance-3 contractions, and
proposes approximate logical-probability gradients at larger distances. Thus
learning logical loss or retaining a logical index is not our novelty. The
potential contribution is a small reusable interface preserving useful update
directions under composition and drift, at measured lower cost.

## 6. What qualifies as progress toward SOTA

There are three separate result levels:

1. A directional development win against our own matched baseline.
2. A frozen, full-refit result on a declared external/public benchmark protocol.
3. A matched SOTA comparison including strong available decoders, observations,
   tuning budget, logical metric, and complete resource accounting.

Do not skip levels. Required comparator families include strong tuned
conventional decoding, DGR-style adaptation, a compatible
[Neural MWPM](https://arxiv.org/abs/2601.00242) implementation, and exact/strong
teachers where tractable. Compare published settings before claiming a win over
a paper; ports to a new drift benchmark are labeled ports.

Proposed progression gates retain the previous >=80% restricted-teacher recovery
and >=70% held-cutpoint reference-gap recovery, with uncertainty supporting the
claim and strict null/shift harm limits. A sheaf arm additionally must beat its
matched non-sheaf relation controls. Better scaling, FPGA footprint, or tail
latency can be contributions without lowest absolute LER, but only on aligned
workloads/hardware and declared noninferiority margins. A Python record latency
cannot be compared directly with AlphaQubit 2's accelerator per-cycle timing.

No search procedure guarantees SOTA. Its useful guarantee is reproducible
selection and honest accounting of which architectural changes help, fail,
or are unidentifiable at the available sample budget.
