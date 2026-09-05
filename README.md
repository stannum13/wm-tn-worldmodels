# Reduced-State and Tensor-Network World Models

Falsification-first study of learned open-system dynamics on public experimental data,
following the plan in `docs/reduced-state-tn-worldmodels.docx`.

**Core object:** a controlled process tensor — a map from a history of interventions to
future reduced states and multi-time observables. The learned model maintains a compressed
predictive state `z_{t+1} = F(z_t, rho_t, a_t)` with `rho_hat_{t+1} = G(z_{t+1})`.

## Current experiment (A) — real-device process world model

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
./scripts/fetch_data.sh          # clone pt_recovery and NMN-tomo under data/external/
python -m pytest tests -q        # unit tests
python scripts/train.py --config configs/exp_a_baselines.yaml
python scripts/evaluate.py --config configs/exp_a_baselines.yaml
```

## Status

- [x] Plan ingested (docx), data sources pinned
- [ ] Experiment A: ingestion, splits, four baselines
- [ ] Directional residual analysis (bias, idle duration, control family, horizon)
- [ ] Iterative rollout-based RL on real sequences (per-episode updates, not batch)
