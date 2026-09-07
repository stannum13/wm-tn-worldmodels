# Reduced-State and Tensor-Network World Models

Falsification-first study of learned open-system dynamics on public experimental data,
following the plan in `docs/reduced-state-tn-worldmodels.docx`.

The current research rationale is in [Why explore world models for quantum
control?](docs/world-models-quantum-control-background.md). The
[redistributed experiment plan](docs/redistributed-experiment-plan.md) replaces the
original RB-gated sequence with independent tests of predictive memory, planning and
policies, streaming observations and response latency, experimental access, physical
constraints, and compression. Its new experiments are planned, not yet executed.

**Core object:** a controlled process tensor — a map from a history of interventions to
future reduced states and multi-time observables. The learned model maintains a compressed
predictive state `z_{t+1} = F(z_t, rho_t, a_t)` with `rho_hat_{t+1} = G(z_{t+1})`.

## First runnable controls

The original plan is broad; see [`docs/process-critique.md`](docs/process-critique.md)
for the process critique. Before fetching tomography data, run:

```bash
PYTHONPATH=src python scripts/run_directional_experiments.py --seeds 10
PYTHONPATH=src pytest -q
```

Add `--output results/directional_controls.json` to save per-seed values and the
5th–95th percentile seed interval.

These controls test known action-memory and physical Bloch-ball effects. They are
implementation checks, not claims about a real device.

## Real benchmark

After fetching the raw `pt_recovery` release, run the length-extrapolation benchmark:

```bash
python scripts/run_pt_recovery_benchmark.py \
  data/external/pt_recovery/experiment_data/RB_data_20230104/len40/idle100/rb_data_0.1/standard_rb_1q_full_data.json \
  --output results/pt_recovery_rb_01.json
```

The current result and its comparison to the published OQE/process-tensor work are in
[`docs/real-benchmark-report.md`](docs/real-benchmark-report.md). The architectural
analysis in [`docs/architecture-findings.md`](docs/architecture-findings.md) shows
that validation-gated shrinkage of the released OQE correction improves the specific
length-41–60 survival-probability forecast. It is not a claim about every process-
tensor observable.

The subsequent [matched memoryless benchmark](docs/memoryless-benchmark-findings.md)
finds that a compact coherent-plus-damping qubit model and a general Markov CPTP
channel explain the active `idle100` sequence-forecast gain without persistent state.
That gain does not transfer to `idle180`; increasing pure-unitary OQE memory through
D6 does not rescue it. See the [independent scientific review](docs/scientific-review.md)
for the revised identification question and experimental controls.

The [delay-aware streaming study](docs/streaming-benchmark-report.md) finds a bounded
favorable regime, then shows in a fresh eight-setting parameter-family test that the
advantage is not generally above 20% and can reverse under low signal and frequent
detector artifacts. This synthetic boundary result motivates a robust-emission model;
it is not a general quantum-feedback claim.

## Original experiment (A) — real-device process world model

Data: `guochu/pt_recovery` (superconducting-qubit randomized-benchmarking sequences with
correlated noise, recovered process tensors) and `Christina-Giar/NMN-tomo` (multi-time
process tomography on a superconducting qubit).

Models under comparison at matched parameter count:

1. time-homogeneous Markov channel,
2. transfer-tensor / linear autoregression,
3. GRU (and a small Transformer variant),
4. causality/CPTP-constrained process-MPO with bond dimension chi.

Split discipline (from the plan): train on sequence length <= 20 and a subset of bias
settings; test on length 40/60, held-out bias, and held-out control families. Random
time-point splits are forbidden — they leak the same physical trajectory.

## Repository interface

Every model implements one contract (`src/ptwm/api.py`):

- `observe` — density matrix, local marginals, or measurement outcomes,
- `act` — pulse, channel, Hamiltonian/quench parameter, or geometry update,
- `latent` — recurrent vector, ADO stack, or MPO bond,
- `predict` — next RDM plus selected multi-time observables,
- `check` — positivity, trace, causality/CPTP residual, N-representability relaxations,
  conservation,
- `budget` — bond/latent dimension, runtime, peak memory.

## Layout

```
src/ptwm/        package: data loading, models, metrics, splits
scripts/         entry points (fetch data, train, evaluate, figures)
configs/         experiment configs (YAML)
tests/           unit + integration tests
results/         metrics tables and figures (small artifacts, tracked)
data/            fetched third-party data (gitignored; see scripts/fetch_data.sh)
docs/            the plan document
```

## Quickstart

```bash
./scripts/fetch_pt_recovery.sh
PYTHONPATH=src python -m pytest tests -q
python scripts/compare_released_oqe.py data/external/pt_recovery \
  --output results/released_oqe_comparison.json
python scripts/train_reconstructed_oqe.py data/external/pt_recovery \
  --bias 0.5 --memory-dimensions 1,2 --seeds 0,1,2 \
  --output results/reconstructed_oqe_bias_05.json
python scripts/run_markov_model_comparison.py data/external/pt_recovery \
  --condition idle100 --biases 0.4,0.5,0.52,0.54 \
  --models damped_d1,markov_cptp --seeds 0,1,2,3,4 --epochs 50 \
  --output results/markov_models_idle100_active.json
```

## Status

- [x] Plan ingested (docx), data sources pinned
- [x] Experiment A: ingestion, leakage-safe splits, and four baseline cells (see `RESULTS.md`)
- [x] Directional residual analysis across bias and horizon
- [x] Iterative rollout-based RL on real sequences (per-episode updates, not batch)
- [x] Released OQE forecast comparison and memory-dimension sweep
- [x] Clifford group recovery and independent differentiable OQE reconstruction
- [x] Matched memoryless-channel comparison and idle-duration transfer test
- [x] Independent `idle180` check (active mixing failed to transfer)
- [ ] NMN-tomo process-matrix physicality residuals (loader present, analysis pending)
- [ ] Acquisition-block uncertainty and prospective active-mixing confirmation
- [ ] Matched planning/policy comparisons in simulations with explicit observation access
- [x] First hidden-detuning streaming/delay screen on GCP
- [x] Fresh eight-setting streaming parameter-family confirmation
- [ ] Robust-emission streaming model and quantum-trajectory extension
- [ ] Observation-design, constraint, and compression experiments from the redistributed plan
