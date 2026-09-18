# Quantum Process-Tensor World Models

Can a compact memory model predict—and eventually help control—how a real quantum device behaves after a long sequence of interventions?

This repository studies that question with public superconducting-qubit data, controlled synthetic systems, and deliberately strong baselines. It spans process tensors, matrix product operators, recurrent world models, differentiable quantum channels, online learning, and delay-aware streaming inference. The project is built around a simple standard: a model is interesting only when its advantage survives leakage-safe splits, matched controls, and physicality checks.

## What is here

- A shared Python pipeline for Markov channels, transfer tensors, process MPOs, GRUs, and Transformers.
- Real-device sequence forecasting across four experimental conditions and 109,200 randomized-benchmarking sequences per condition.
- Independent reconstruction and auditing of the released observable quantum embedding (OQE) models.
- Matched coherent, damping, and general CPTP baselines that test whether apparent memory really requires a stateful model.
- Online rollout experiments with a learned gate policy and an equivalent batch control.
- Delay-aware streaming world models with detector artifacts, latency, and fresh parameter-family tests.
- Direct physicality and quantum-memory analysis of multi-time process matrices.
- Reproducible JSON results, plots, research notes, and synthetic test coverage.

## Results at a glance

| Question | What the experiments found |
|---|---|
| Can a small temporal-memory model improve long-horizon forecasts? | Yes, in a bounded regime. A 79-parameter process MPO beats the Markov baseline on both idle-100 ns horizon tests; its largest log-MSE improvement is 44%. |
| Does that establish persistent quantum memory? | No. Stronger coherent-plus-damping and general Markov CPTP controls explain the active idle-100 forecast gain without persistent hidden state. |
| Does more memory capacity reliably help? | No. Larger MPO/OQE memory does not improve monotonically, and increasing released OQE memory through D6 does not rescue transfer to idle 180 ns. |
| Did online learning beat an equivalent batch model? | No. Once both use identical normalized features, the batch model has lower MAE in all four cells. The initial apparent gain was a feature mismatch. |
| Is streaming memory useful under latency and artifacts? | Sometimes. A bounded favorable regime exists, but a fresh eight-setting test shows the advantage can fall below 20% or reverse at low signal and high artifact rates. |
| Do the tomography artifacts contain a quantum-memory signal? | Yes. The physical process matrices reproduce the source experiment's nonzero partial-transpose negativity (mean 0.0065, max 0.0217). |

The result is more useful than a blanket “world models win” story: this code maps where memory helps, where a simpler physical model explains the same behavior, and where a promising effect fails to transfer.

## Research map

- [Experiment A results](RESULTS.md) — model ladder, structural splits, residuals, rollout learning, and NMN-tomo physicality.
- [Why world models for quantum control?](docs/world-models-quantum-control-background.md) — motivation and background.
- [Scientific review](docs/scientific-review.md) — revised identification question and missing controls.
- [Real benchmark report](docs/real-benchmark-report.md) — length extrapolation on released device data.
- [Architecture findings](docs/architecture-findings.md) — validation-gated OQE shrinkage and forecast behavior.
- [Memoryless benchmark findings](docs/memoryless-benchmark-findings.md) — matched alternatives to persistent memory.
- [Streaming benchmark report](docs/streaming-benchmark-report.md) — response delay, artifacts, and robustness boundaries.
- [Redistributed experiment plan](docs/redistributed-experiment-plan.md) — independent tests of memory, planning, observations, constraints, and compression.

## Experimental design

The main real-device dataset is [guochu/pt_recovery](https://github.com/guochu/pt_recovery), containing superconducting-qubit randomized-benchmarking sequences under correlated noise. The original experiment trains on sequence lengths up to 20, then tests longer horizons, unseen noise-bias settings, and a held-out Clifford family. Random time-point splits are avoided because they leak the same underlying trajectory.

The held-out-family split is retained as a negative-control lesson: gate-family membership is confounded with length in this release, so it cannot support a clean compositional-generalization conclusion.

The second dataset, [Christina-Giar/NMN-tomo](https://github.com/Christina-Giar/NMN-tomo), provides experimental, physical-projected, and Markovian multi-time process matrices. The analysis keeps two quantities separate:

- negative eigenvalues of the raw reconstruction measure failure of positive semidefiniteness; and
- partial-transpose negativity of the normalized physical matrix measures the quantum non-Markovian signal reported by the source experiment.

Synthetic directional and streaming systems are used as implementation and stress tests, not presented as real-device evidence.

## Model and evaluation flow

~~~text
controls + observations + device context
                    |
                    v
 Markov / transfer tensor / process MPO / GRU / Transformer
                    |
                    v
       state or fidelity forecast
                    |
                    v
 horizon | transfer | latency | residual | physicality checks
                    |
                    v
 matched memoryless and batch controls
~~~

All sequence models are causal: future gates are never visible. The process MPO uses a contractive latent transition, while bond dimension exposes the accuracy-versus-complexity tradeoff.

## Reproduce the work

Python 3.10 or newer is required.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
./scripts/fetch_data.sh
python -m pytest
~~~

Run one original experiment cell and its analyses:

~~~bash
python -m ptwm.run_experiments --length-cap 40 --idle 100 --out results
python -m ptwm.residuals --length-cap 40 --idle 100 --out results
python -m ptwm.rl --length-cap 40 --idle 100 --out results --n-episodes 5000
python -m ptwm.nmn --root data/external/NMN-tomo --out results
~~~

Run the newer controls and benchmarks:

~~~bash
PYTHONPATH=src python scripts/run_directional_experiments.py --seeds 10
PYTHONPATH=src python scripts/run_pt_recovery_benchmark.py \
  data/external/pt_recovery/experiment_data/RB_data_20230104/len40/idle100/rb_data_0.1/standard_rb_1q_full_data.json \
  --output results/pt_recovery_rb_01.json
PYTHONPATH=src python scripts/run_markov_model_comparison.py data/external/pt_recovery \
  --condition idle100 --biases 0.4,0.5,0.52,0.54 \
  --models damped_d1,markov_cptp --seeds 0,1,2,3,4 --epochs 50 \
  --output results/markov_models_idle100_active.json
PYTHONPATH=src python scripts/run_streaming_confirmation.py
~~~

Datasets are downloaded into the ignored data/external/ directory. Small derived artifacts are tracked so the reported results remain inspectable without rerunning every training job.

## Repository guide

| Path | Purpose |
|---|---|
| src/ptwm/models.py | Original Markov, transfer-tensor, process-MPO, GRU, and Transformer ladder |
| src/ptwm/oqe.py | Differentiable observable quantum embedding models |
| src/ptwm/streaming.py | Causal streaming estimators and delay-aware evaluation |
| src/ptwm/real_benchmark.py | Released-device benchmark support |
| src/ptwm/rl.py | Online transition learning, policy rollouts, and matched batch controls |
| src/ptwm/nmn.py | Process-matrix physicality and partial-transpose negativity |
| scripts/ | Reproducible benchmark and analysis entry points |
| results/ | Machine-readable metrics and generated figures |
| tests/ | Synthetic unit and end-to-end tests |
| docs/ | Data notes, experiment reports, critique, and research rationale |

## Scope and limitations

This is a research codebase, not a production quantum-control stack. Several neural tables still need broader seed sweeps; sequence fidelity is a compressed observable rather than a full state; the family holdout is length-confounded; streaming conclusions currently come from controlled synthetic systems; and the rollout reward is not stable enough to support a policy-improvement conclusion.

Those limits are part of the work. The objective is to separate genuine temporal-memory value from optimization effects, feature choices, simulator advantages, and ordinary Markov physics.
