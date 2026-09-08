# Graph-overlay compiler research program

Decision note, 8 September 2026.

## Central thesis

Decoder adaptation should be organized around bounded graph operations:

\[
G_t=G_0\oplus G_t^{\mathrm{state}}\oplus
G_t^{\mathrm{correlation}}\oplus G_t^{\mathrm{hypothesis}}.
\]

Here \(G_0\) is an immutable calibrated graph or factor graph. The overlay symbol is
typed composition, not an unrestricted algebraic sum: each overlay may change only
declared slots, motifs, regions, or plan identifiers; composition order and conflicts
are resolved at compile time. Runtime code cannot discover arbitrary topology.

The unifying scientific hypothesis is:

> QEC decoder failures may admit a small taxonomy of local graph defects—incorrect
> potentials, hidden modes, missing factors, insufficient hypothesis width, and an
> insufficient solve region—and a compiled controller can repair those defects online
> without replacing the trusted constraint-preserving decision layer.

This is broader than “memory around MWPM” but narrower than an end-to-end learned
decoder. Individual operations have precedents; the proposed contribution is to test
whether defect-directed selection produces a better accuracy/resource/stability
frontier under one matched observation contract.

## Typed instruction set

| defect | legal operation | runtime freedom | predicted failure boundary |
|---|---|---|---|
| calibrated mechanisms, wrong rates | `REWEIGHT(edge_class, bounded_delta)` | update predeclared potentials | cannot represent missing correlations |
| persistent leakage/reset/readout mode | `ACTIVATE_MODE(site, plan_id)` | select/marginalize a local template | fails if observations do not identify the mode |
| known correlated mechanism omitted | `INSERT_FACTOR(motif_id)` / `REWIRE(motif_id)` | activate a compiled factor or equivalent local rewrite | motif library cannot express novel support |
| two or more credible local explanations | `FORK(region, hypothesis_ids, K)` | duplicate only a bounded local region | global or high-width ambiguity exceeds budget |
| BP/inversion failure localized in space | `GROW_REGION(region, radius)` and `LOCAL_SOLVE(region, solver_id)` | expand to a fixed cap and invoke a bounded solver | above-threshold percolation destroys locality |
| observational aliasing | `ADD_OBSERVATION(factor_id)` | request/use a preauthorized measurement or metadata field | passive replay cannot establish intervention value |
| interconnect or SRAM bottleneck | `PARTITION(graph, hardware_map)` | select a compiled placement | dynamic traffic may violate the assumed cut budget |

EMA, CUSUM, HMM beliefs, contractive state-space filters, and small KAN/MLP teachers
are trigger implementations. They are not the primary instruction set.

All hot-path programs obey:

- immutable base topology and code constraints;
- bounded probability/weight changes and saturated state;
- finite motif, mode, region, and solver libraries;
- no allocation, recursion, learned addresses, or unbounded loops;
- atomic publication only between records;
- fail-closed fallback to \(G_0\);
- exact replay or simulator validation before archive admission.

## What the reduced world model compiles

The reduced model maps causal evidence to a vector of *intervention values*, not
directly to an error label:

\[
o_{1:t}\longrightarrow
(\Delta\ell_{\rm reweight},\Delta\ell_{\rm mode},\Delta\ell_{\rm factor},
\Delta\ell_{\rm fork},\Delta\ell_{\rm local\ solve}).
\]

For decoder-only replay, each component is estimated by applying the corresponding
legal overlay to the same fixed record and scoring the paired terminal logical loss
plus resource cost. Training only on failed records would induce selection bias and
hide operations that harm previously correct decisions. Therefore label every record
or independent episode, retain harmful as well as helpful outcomes, and include
multi-operation interactions when single operations do not compose additively.

The teacher may cheaply screen candidates, predict future nuisance state, or identify
promising graph regions. It cannot admit its own recommendation. The trusted replay or
simulator evaluates the compiled program end to end.

Search begins with enumeration and Pareto/quality-diversity methods because the grammar
is small and discrete. RL is reserved for genuinely sequential compilation decisions,
such as region growth, hypothesis-budget allocation, or active observation selection.
The archive objective remains a vector:

\[
(\mathrm{LER},p50,p99,\mathrm{WCET},\mathrm{energy},\mathrm{LUT},\mathrm{BRAM},
\mathrm{state\ bytes},\mathrm{instability},\mathrm{escalation}).
\]

Deadlines, state caps, contraction, and legal topology are hard constraints.

## Defect-isolating benchmark ladder

Do not begin with a mixture of every failure mode. Construct paired positive and
negative controls in which exactly one operation should have an oracle advantage:

| arm | injected defect | oracle operation | critical negative control |
|---|---|---|---|
| O0 | stationary calibrated independent noise | none | every overlay must be harmless |
| O1 | local rate drift | `REWEIGHT` | shuffled/lag-destroyed history |
| O2 | persistent observable leakage/reset mode | `ACTIVATE_MODE` | emission made uninformative |
| O3 | known 3+ detector correlated event | `INSERT_FACTOR` | motif absent or wrong support |
| O4 | two locally plausible explanations | `FORK(K=2)` | unimodal matched-entropy case |
| O5 | localized BP trapping/oscillation | `GROW_REGION` + `LOCAL_SOLVE` | above-threshold percolating region |
| O6 | aliased O2 | `ADD_OBSERVATION` | same sensing/query budget without the added factor |
| O7 | growing multi-tile graph | `PARTITION` | randomized placement with matched compute |

Each arm reports static-to-oracle opportunity, teacher recovery, compiled recovery,
stationary harm, latency/resource cost, and cross-arm false activation. A primitive
advances only if it recovers at least 80% of its oracle gain, has a positive paired
interval for the logical endpoint, harms O0 by no more than 0.10 percentage points,
and does not activate materially more often in the wrong-defect controls. The 80% and
0.10-point thresholds are frozen program choices, not literature facts.

The current tiny repetition-code pilot covers only O1/O2 using an exact two-class
repetition constraint layer. It cannot test missing factors, rewiring, local
hypothesis lifting, qLDPC BP failure, observation value, or hardware partitioning.

## Literature boundary

- [DGR](https://arxiv.org/abs/2311.16214) establishes adaptive graph reweighting from
  matching statistics; `REWEIGHT` itself is not novel.
- [AlphaQubit](https://www.nature.com/articles/s41586-024-08148-8) establishes that
  per-stabilizer recurrent state plus spatial/global mixing can exploit richer noise;
  it motivates the teacher and local/global ablation.
- [HyperBlossom](https://arxiv.org/abs/2508.04969) formulates decoding directly on
  parity-factor hypergraphs; it motivates keeping correlated motifs as factors rather
  than pretending every event is pairwise.
- [GARI](https://www.nature.com/articles/s41467-026-70556-3) modifies qLDPC graph
  structure to improve inference under correlated errors.
- [Beam search decoding](https://journals.aps.org/prxquantum/abstract/10.1103/6k5x-ztqt)
  motivates explicit hypothesis-budget tradeoffs, but does not by itself establish
  local hypothesis lifting.
- [Localized statistics decoding](https://www.nature.com/articles/s41467-025-63214-7)
  identifies, grows, and solves local decoding regions; `GROW_REGION` and
  `LOCAL_SOLVE` must be evaluated against it rather than presented as new primitives.
- [HeliosNet](https://ieeexplore.ieee.org/document/10821062/) demonstrates why graph
  partition and inter-device communication are part of the decoder architecture.

Accordingly, the plausible novelty is a typed overlay compiler, failure-attribution
protocol, local hypothesis lift, and matched Pareto evaluation—not any isolated graph
operation.

## Claim boundary

Counterfactual graph replay is valid when a decoder output cannot change later logged
measurements. `ADD_OBSERVATION`, active reset, pulse selection, and other feedback do
change the future and require an interactive simulator or prospective hardware.
Existing Rigetti files are repeatedly accessed development data and cannot supply the
fresh-session confirmation for this program.
