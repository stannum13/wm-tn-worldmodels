# Data notes (pinned schemas)

## pt_recovery (primary, Experiment A)

Source: https://github.com/guochu/pt_recovery (paper: "Learning and forecasting open
quantum dynamics with correlated noise", Commun. Phys. 2025). Cloned by
`scripts/fetch_data.sh` into `data/external/pt_recovery/`.

Layout under `experiment_data/RB_data_20230104/`:

```
{len40,len60}/{idle100,idle180}/rb_data_{g}/data.json
```

- `data.json`: `{"train_data": [[[g1..gL], f], ...] (4680 entries),
  "test_data": [...] (3120 entries)}`.
- One episode = a single randomized-benchmarking sequence: `g_i` are gate indices into a
  24-element single-qubit Clifford set (values observed in 0..23), `f` is the measured
  unitary fidelity of the whole sequence against the ideal Clifford (float, may slightly
  exceed 1 due to measurement noise; observed range ~0.61..1.015).
- Lengths L = 2..40 (len40 dirs) or 2..60 (len60 dirs), 120 train + 80 test shots per
  length per bias folder.
- Bias axis: folder suffix `g` in {0.1, 0.2, 0.3, 0.4, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6,
  0.61, 0.62, 0.63, 0.64}; actual Vbias = 0.4 * g (per repo README).
- Idle axis: `idle100` vs `idle180` (idle duration in ns between gates).
- `rb_data_{g}.npy` (39,): smoothed mean fidelity per length (2..40); a pre-averaged
  variant of the same signal (corr with shot-mean ~0.85; different averaging).
- `unitaryprocesstensor_D*.json`: Julia `Serialization` blobs (not JSON) — the paper's
  reconstructed process tensors at bond dimension D. Not readable from Python; we use the
  raw episodes and the paper's published result JSONs instead.
- `result/*.json`: paper outputs. `RB_data_len40_idle{100,180}_unitary_results.json`:
  `{Ds: [1..6], gammas: [14 biases], fidelities/test_fidelities: [D][gamma][length]}` —
  reconstruction fit errors per bond dimension. `predict_60_full_data_D*.json`: paper's
  len-60 predictions per D. Used only as reference points for our baselines.

## NMN-tomo (secondary)

Source: https://github.com/Christina-Giar/NMN-tomo (paper: "Multi-time quantum process
tomography on a superconducting qubit", Quantum 2025).

- `NMN_lab_rslts.json`: `{detuning_key: {prep_meas_basis: {outcome: counts}}}` for the
  lab-frame dataset (1 detuning point).
- `NMN_tomog_rerun.json`: same structure for a 3x3 grid of detuning points
  (keys like "21.333,24.889"), each with 4 Pauli-prep/meas settings x 4 outcome counts
  (~8000 shots each).
- `Ws/Wexp_all.mat`, `Wphys_all.mat`, `Wmark_all.mat`: reconstructed multi-time process
  matrices (experimental / physical-projected / Markovian) as .mat — usable from Python
  via `scipy.io.loadmat` for physicality-residual reference values.

## Modeling consequences

- The published pt_recovery observables are sequence fidelities, not density matrices.
  The RDM-level `check` (positivity/trace) therefore reduces, in this dataset, to
  observable-space physicality: predicted fidelity in [0, 1 + noise allowance], and
  causality = contractivity of the learned temporal map (spectral norm <= 1 for the
  process-MPO latent transition). Full CPTP residuals are computed on the NMN-tomo
  process matrices, where actual multi-time process matrices exist.
- Model ladder on a common axis (temporal memory): Markov channel (no memory) ->
  transfer tensor (scalar exponential kernel) -> process-MPO (learned low-bond linear
  latent, chi in {1,2,4,8}) -> GRU (nonlinear latent). All predict log-fidelity; all
  sequential; none sees future gates (causal by construction).
