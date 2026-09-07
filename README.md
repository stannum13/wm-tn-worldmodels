# World models for quantum dynamics and control

This repository asks: **when does a learned predictive state improve quantum-system
inference or control enough to justify its measurement, training, and response-time
costs?**

The work combines public superconducting-qubit data with auditable simulations. It
does not treat a history-dependent prediction gain as proof of physical quantum
memory. Models are compared with Markov, classical hidden-state, direct-search, and
causal-filtering baselines under matched observation access.

## Current evidence

- The public `pt_recovery` randomized-benchmarking data have been reproduced across
  four length/idle cells; see [Experiment A](RESULTS.md).
- A compact coherent-plus-damping model and a general Markov CPTP channel explain the
  active `idle100` forecast gain without persistent memory. It does not transfer to
  `idle180`; see the [memoryless benchmark](docs/memoryless-benchmark-findings.md).
- A synthetic streaming screen found favorable delay-aware forecasting conditions,
  but an eight-setting test did not establish a general 20% advantage and exposed
  failures under low signal and detector artifacts; see the
  [streaming report](docs/streaming-benchmark-report.md).
- Multi-rate causal heads are being evaluated as slow-path context models. A held
  output feeds a cheap hot-path denoiser; KAN-inspired spline heads must beat equally
  scheduled linear heads on both error and measured amortized latency.
- The first 31-parameter spline head is a NO-GO: it adds at most about 1% over the
  6-parameter linear head while costing 6–8 times more per update. See the
  [multirate architecture note](docs/multirate-causal-architecture.md).
- Particles are a NO-GO for an ordinary nonlinear unimodal stream but a GO as a slow
  inference lane when wrapped observations make the posterior genuinely multimodal;
  64 particles recover 91% of the EKF-to-grid loss gap. See the
  [wrapped-phase report](docs/wrapped-phase-particle-report.md).
- On a chronological holdout of real Ankaa-2 I/Q calibration shots, a 13-parameter
  spline/KAN logistic head reduces Brier loss 6.0% versus a matched affine logistic
  head, without an established classification-error gain. See the
  [real I/Q report](docs/rigetti-real-iq-report.md).
- That calibration gain does not survive the locked 100,000-shot stability-9 logical
  replay. Independent soft-parity propagation is a NO-GO; graph-aware joint evidence
  is now required. See the [QEC replay report](docs/rigetti-qec-replay-report.md).
- Static circuit-local pair and cumulative-parity expansions also fail to close the
  gap to released MWPM. See the
  [graph-feature report](docs/rigetti-graph-feature-report.md).
- A 31-parameter categorical syndrome model recovers 14.1% of that gap, but adding
  temporal Markov order hurts. This is below the 20% gate and redirects work to a true
  matching baseline. See the
  [Markov decoder report](docs/rigetti-markov-decoder-report.md).
- A transparent uniform circuit-noise model plus PyMatching reaches 38.7475%, matching
  the released 38.819% stability-9 anchor to within 0.0715 percentage points. Learned
  components must now improve this structural control. See the
  [matching report](docs/rigetti-matching-control-report.md).

These are bounded prediction findings, not closed-loop hardware-control claims.

## Research structure

The [background note](docs/world-models-quantum-control-background.md) explains how
world models might help as predictors, planners, policies, belief states, and probe
selectors. The [experiment plan](docs/redistributed-experiment-plan.md) defines
observation contracts, mechanism families, quantitative targets, and GO/NO-GO rules.
The [process critique](docs/process-critique.md) and
[scientific review](docs/scientific-review.md) document limitations.
The [autonomous audit](docs/autonomous-science-audit.md) records corrections made after
the streaming runs, and the [deployment ladder](docs/deployment-benchmark-ladder.md)
prioritizes public real-hardware datasets.

The core predictive object is a compressed causal state:

```text
z[t+1] = F(z[t], observation[t], action[t])
prediction[t+1] = G(z[t+1])
```

Simulator-only states are privileged references, never free policy observations.
Streaming claims use only data available at the decision timestamp.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Fetch public data with `./scripts/fetch_data.sh`. Raw third-party data are stored in
`data/external/` and are not committed.

## Principal commands

```bash
# Cheap implementation controls
PYTHONPATH=src python scripts/run_directional_experiments.py \
  --seeds 10 --output results/directional_controls.json

# Matched real-data Markov comparison
PYTHONPATH=src python scripts/run_markov_model_comparison.py \
  data/external/pt_recovery --condition idle100 \
  --biases 0.4,0.5,0.52,0.54 --models damped_d1,markov_cptp \
  --seeds 0,1,2,3,4 --epochs 50 \
  --output results/markov_models_idle100_active.json

# Streaming delay and fresh parameter-family screens
PYTHONPATH=src python scripts/run_streaming_benchmark.py \
  --output results/streaming_benchmark.json
PYTHONPATH=src python scripts/run_streaming_confirmation.py \
  --workers 8 --output results/streaming_confirmation.json

# Multi-rate causal heads
PYTHONPATH=src python scripts/run_causal_head_benchmark.py \
  --seeds 10 --delay 25 --strides 1,8,32,128 \
  --output results/causal_head_benchmark.json

# Nonlinear EKF/particle feasibility screen
PYTHONPATH=src python scripts/run_nonlinear_filter_benchmark.py \
  --seeds 10 --delay 10 --output results/nonlinear_filter_benchmark.json

# Multimodal wrapped-phase positive control
PYTHONPATH=src python scripts/run_wrapped_phase_benchmark.py \
  --seeds 10 --streams 30 --length 800 --delay 10 \
  --particles 32,64,128,256 --output results/wrapped_phase_benchmark.json

# Real Ankaa-2 I/Q calibration (requires: pip install -e '.[real]')
./scripts/fetch_rigetti_fast_feedback.sh
PYTHONPATH=src python scripts/run_rigetti_iq_benchmark.py \
  --output results/rigetti_iq_benchmark.json

# Logical replay on the 100,000-shot stability-9 file
./scripts/fetch_rigetti_stability9.sh
PYTHONPATH=src python scripts/run_rigetti_qec_replay.py \
  --data data/rigetti_stability9/stability_9_raw_data.h5 \
  --circuit-group circuit_26 --output results/rigetti_stability9_qec_replay.json

# Matching control using an explicit approximate circuit-noise model
PYTHONPATH=src python scripts/run_rigetti_matching_control.py \
  --data data/rigetti_stability9/stability_9_raw_data.h5 \
  --circuit-group circuit_26 --output results/rigetti_matching_control.json
```

The causal-head benchmark currently uses delayed simulator-state targets as a
privileged diagnostic. It cannot support a deployable learned-policy claim until an
experimentally accessible delayed verification signal replaces those targets.

## Repository map

```text
src/ptwm/   loaders, models, splits, metrics, and streaming primitives
scripts/    reproducible experiment entry points
tests/      unit and public-data integration tests
results/    versioned metrics, predictions, and figures
docs/       rationale, critiques, plans, and evidence reports
data/       fetched third-party data; ignored by Git
```

## Campaign status

- [x] Public-data ingestion, leakage-aware splits, and Experiment A baselines
- [x] Released/reconstructed OQE and matched memoryless comparisons
- [x] Idle-duration transfer test
- [x] Streaming delay screen and eight-setting parameter-family test on GCP
- [x] Multi-rate linear/KAN-inspired causal-head diagnostic on GCP
- [x] Nonlinear robust-EKF/particle feasibility screen on GCP
- [x] Wrapped-phase multimodal particle positive control on GCP
- [x] Real Ankaa-2 I/Q chronological calibration screen on GCP
- [x] Real Ankaa-2 100,000-shot logical soft-decoding NO-GO on GCP
- [x] Reproduce released stability-9 MWPM scale with a transparent matching control
- [ ] Robust contamination-aware emission model
- [ ] Backaction-consistent quantum-trajectory control benchmark
- [ ] Matched planning/policy comparison with explicit observation costs
- [ ] NMN-tomo process-matrix physicality analysis
- [ ] Prospective hardware or held-out real-deployment confirmation

Autonomous campaign changes are in draft
[GitHub pull request #1](https://github.com/stannum13/wm-tn-worldmodels/pull/1).
