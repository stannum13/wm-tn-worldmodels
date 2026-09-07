# Experiment A results — real-device process world model

Data: `guochu/pt_recovery` superconducting-qubit RB sequences with correlated noise
(4 cells: sequence cap {40, 60} x idle {100, 180} ns x 14 bias settings; 109,200
episodes per cell). Splits are structural per the plan: horizon (train L <= 20,
test L > 20), held-out bias (0.61, 0.63), held-out gate family (Clifford indices
16-23 absent from training). No random time-point splits anywhere.

All models are bias-conditioned and causal (no future gates visible). Primary
metric: MSE in log-fidelity space. Reproduce with:

```bash
./scripts/fetch_data.sh
python3.10 -m ptwm.run_experiments --length-cap 40 --idle 100 --out results
python3.10 -m ptwm.residuals   --length-cap 40 --idle 100 --out results
python3.10 -m ptwm.rl          --length-cap 40 --idle 100 --out results --n-episodes 5000
python3.10 -m ptwm.aggregate   --results results
```

## 1. Baselines (best model per cell/split, log-MSE; Markov channel in parens)

| cell | horizon split | held-out bias | held-out family |
|---|---|---|---|
| len40/idle100 | **process-MPO chi2** 0.128 (0.152) | **process-MPO chi2** 0.165 (0.173) | transfer-tensor 0.200 (0.229) |
| len40/idle180 | transfer-tensor 0.096 (0.126) | GRU-8 0.086 (0.096) | transfer-tensor 0.165 (0.429) |
| len60/idle100 | **process-MPO chi2** 0.090 (0.161) | **process-MPO chi4** 0.110 (0.115) | transformer 0.200 (0.293) |
| len60/idle180 | transfer-tensor 0.079 (0.101) | **process-MPO chi2** 0.093 (0.098) | transfer-tensor 0.189 (0.258) |

Findings:

- **H1 is supported on the horizon and bias directions, conditionally.** The
  79-parameter causality-constrained process-MPO wins the horizon split on both
  idle100 cells (largest margin: len60/idle100, 44% below the Markov channel) and
  the held-out-bias split on 3 of 4 cells. It never wins by a large margin on
  idle180 cells, where the 6-parameter transfer tensor (a scalar exponential
  memory kernel) is sufficient — longer idle appears to wash out the correlated
  gate-level structure the MPO bond captures.
- **Bond dimension does not monotonically help**: chi=4 beats chi=2 only once
  (len60/idle100 bias split). Consistent with the plan's warning that bond
  dimension is a budget, not a target.
- **The family split is confounded with length** (sequences avoiding an 8-gate
  family are almost all short), so its numbers measure short-sequence training
  plus length shift, not gate-family generalization. Treat that column as a
  negative-control illustration of split design, not evidence about models.

## 2. Directional residuals

Decomposition of test residuals by forecast horizon, bias, gate family, and
(bias x length) context cell — `results/residuals_len*_idle*.json`,
per-length curves in `results/residuals_by_length_*.png`.

- **Markov-channel failure is systematic, not random.** On len60/idle100, 86% of
  the Markov channel's residual variance is explained by (bias, length) cell
  means (len40/idle100: 32%). Its per-length residual drifts to -0.25 log-fidelity
  by L=40 — a pure extrapolation bias.
- **The process-MPO absorbs that structure**: only 15-20% of its residual variance
  is context-explained, and its per-length residual is flat (-0.03 to -0.05) out
  to L=40. The low-bond latent is doing exactly the job the plan predicts for a
  compressed temporal memory.
- Residuals by bias show the same pattern: Markov errors grow monotonically with
  bias (up to -0.35 at gamma=0.4 on len40/idle100); the MPO's stay within
  +/-0.25 and are roughly flat across the bias axis.

## 3. Iterative rollout RL (per-episode updates, never whole-batch)

Linear-softmax policy over (context, last gate) + linear transition model updated
by SGD after every episode; REINFORCE with running baseline and clipped
advantages; epsilon-greedy exploration decaying 0.2 -> 0.05 over 5,000 episodes.

| cell | online MAE | batch Markov MAE | random-policy control | reward first100 -> last100 |
|---|---|---|---|---|
| len40/idle100 | **0.254** | 0.301 | 0.262 | -0.43 -> -0.28 |
| len40/idle180 | 0.250 | 0.288 | **0.222** | -0.41 -> -0.36 |
| len60/idle100 | **0.236** | 0.339 | 0.248 | -0.38 -> -0.40 |
| len60/idle180 | 0.227 | 0.259 | **0.198** | -0.41 -> -0.43 |

Findings and honest caveats:

- **Per-episode online learning is competitive with — and on 2 of 4 cells better
  than — whole-batch fitting**, despite seeing 5,000 episodes vs the batch
  model's 53,200. Physicality violations stay low (1.6-6.5% of episodes).
- **Attribution caveat**: the random-exploration control shares the online
  model's length-normalized count features, which is where most of the gain over
  the batch Markov channel (raw-count features) comes from. The learned policy's
  own contribution is the online-vs-random margin, which is positive on 2 of 4
  cells only.
- **The policy reward curve is not yet reliable**: it improves on both len40
  cells but declines slightly on both len60 cells. With a linear policy and
  5,000 episodes this is within noise; a rollout-based policy study that claims
  H1-style gains needs longer runs and a matched-feature batch control.

## 4. Threats to validity

- Single seed per cell for the torch models (seed 0); seed sensitivity unquantified.
- The family split's length confound (above).
- The RL comparison conflates feature parameterization (normalized vs raw counts)
  with training regime (online vs batch); a normalized-feature batch control is
  the missing ablation.
- Observable-space physicality (fidelity in [0, 1+eps], contractive latent) is a
  surrogate for CPTP; full process-matrix residuals require the NMN-tomo
  process matrices (loader present, analysis not yet run).

## 5. Artifact map

- `results/exp_a_len{40,60}_idle{100,180}.json` — full baseline metrics per split
- `results/exp_a_summary.csv` — 84-row flat table, all cells
- `results/residuals_len*_idle*.json` + `residuals_by_length_*.png` — directional residuals
- `results/rl_len*_idle*.json` + `rl_curves_*.png` — RL learning curves and eval
- `docs/data_notes.md` — pinned data schemas
