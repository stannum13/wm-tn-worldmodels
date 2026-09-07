# Matched memoryless benchmark and idle-duration transfer

Date: 2026-09-06. Source data commit:
`47d67598a304bb72c315bf05ddfadcda5f4be290`.

## Question

Does prediction of individual RB sequences require a persistent environment after
ordinary coherent error, dissipation, control-axis alignment, and readout calibration
are represented fairly?

This benchmark addresses predictive necessity within tested model classes. It does
not witness or exclude physical quantum memory in the device.

## Models

`damped_d1` is a time-homogeneous coherent qubit error interleaved with exact
Cliffords, isotropic Bloch contraction, a shared learned preparation/measurement
axis in the recovered abstract Clifford representation, and bounded two-outcome
readout calibration. It stores nine parameters, with gauge redundancies.

`markov_cptp` is a general time-homogeneous qubit CPTP channel represented by four
normalized complex Kraus operators, with the same axis alignment and readout model.
It stores 40 parameters, also with Kraus and coordinate gauge redundancies. Its
trace-preservation residual is checked after fitting.

Each model uses five prespecified seeds and at most 50 epochs. A seed is selected by
RMSE on the released `test_data`, which contains a random 40% of sequences at every
length 2–40. Training uses the disjoint released 60%. Final forecasting uses the
independent length-60 acquisition at lengths 41–60. The final outcomes do not select
seeds. This is a matched protocol for these two new models; it is not matched in
compute or partitioning to every released OQE artifact.

## Development condition: idle100

Best seed selected by validation RMSE:

| Bias | RB | Damped D1 | General Markov CPTP |
|---:|---:|---:|---:|
| 0.40 | 0.14067 | **0.08276** | 0.08418 |
| 0.50 | 0.18341 | **0.07807** | 0.07922 |
| 0.52 | 0.16391 | **0.08105** | 0.08142 |
| 0.54 | 0.12939 | **0.08569** | 0.08694 |
| Mean | 0.15435 | **0.08189** | 0.08294 |

Both memoryless models explain most of the sequence-dependent prediction gain in
the four conditions where the earlier residual hybrid was active. Damped D1 is
numerically best in all four, but its average advantage over the larger CPTP model
is only 0.00105 RMSE and has not been assigned acquisition-block uncertainty. Treat
the models as practically tied until that uncertainty is available.

This is stronger than fitting the average RB curve: the prior within-length audit
showed that sequence-specific correspondence supplies the improvement. It is also
consistent with standard physics: a fixed coherent error is conjugated into a
sequence-dependent path by the intervening Clifford controls.

The result rejects a claim that persistent environmental state is needed to obtain
these `idle100` forecast errors within the tested protocol. It does not identify the
fitted contraction or readout parameters as device properties; several effects may
trade off under terminal survival measurements.

## Frozen-method check: idle180

After choosing architecture, restart count, epoch budget, optimizer settings, and
selection method on `idle100`, the same script was run on the previously unexamined
`idle180` condition. Models were retrained on `idle180`; this is method transfer, not
zero-shot parameter transfer.

| Bias | RB | Damped D1 | General Markov CPTP | Validation decision |
|---:|---:|---:|---:|---|
| 0.10 | 0.05730 | 0.05266 | **0.05172** | CPTP improves slightly |
| 0.40 | **0.18372** | 0.18435 | 0.18498 | fall back to RB |
| 0.50 | 0.17221 | **0.17216** | 0.17231 | validation favors RB |
| 0.52 | 0.14742 | **0.14707** | 0.14729 | validation favors RB |
| 0.54 | 0.11482 | **0.11468** | 0.11488 | validation favors RB |
| 0.60 | 0.04265 | **0.04240** | 0.04356 | validation favors RB |

At biases 0.4–0.54, mean RMSE is 0.15454 for RB, 0.15457 for damped D1, and
0.15487 for general CPTP. Small test differences at 0.5–0.54 do not count as model
selection wins because validation already favors RB. Thus the active sequence-model
gain does not transfer. At bias 0.1 the full channel improves RMSE by 0.00558, but
shared-calibration/acquisition uncertainty has not been recovered, so this remains a
conditional predictive result.

Raw and processed length-40 files match exactly as multisets for every checked bias,
ruling out a split-construction mismatch. The regime changes with idle duration:
longer sequences approach a broad distribution near 0.5, and the tested memoryless
models fail to predict the remaining sequence variation at intermediate biases.

## Does a small pure-unitary memory rescue idle180?

At bias 0.5, the independently reconstructed pure-unitary OQE was tested with D1 and
D2 across five seeds. D2 was also tested across initial generator scales 0.05, 0.2,
0.5, and 1.0; no validation run produced a useful correction. D4 and D6 capacity
checks improved monotonically but still lost to RB:

| Model | Best validation RMSE | Its corresponding test RMSE | RB test RMSE |
|---|---:|---:|---:|
| Pure OQE D2, all tested scales | 0.29746 | 0.28145 | 0.17221 |
| Pure OQE D4 | 0.26277 | 0.23933 | 0.17221 |
| Pure OQE D6 | 0.25206 | 0.22431 | 0.17221 |

The old 0.5 hybrid threshold assigns zero OQE weight in every run. This threshold is
not a confidence test, but the conclusion here does not depend on its exact value:
the standalone models and their validation corrections are plainly poor.

A new dissipative D2 implementation with retained, dephased, or reset memory also
collapsed to the mean from small initialization. Ten retained-memory seeds and an
initial multiscale pure-OQE sweep did not solve the optimization. Because dissipation
creates an easy low-variance basin, the retain/dephase/reset comparison is not yet a
valid negative result. It needs a staged or centered-residual objective that first
learns sequence structure, followed by calibration. That method must be developed
on a new development partition; `idle180` has now been consumed by exploration.

A follow-up objective added twice the ordinary weight to errors after centering both
targets and predictions within each sequence length. Across five seeds, neither the
retained D2 model nor a 13-parameter quasi-static Gaussian classical-noise model left
the mean-prediction basin. At bias 0.5 their best validation RMSE remained about
0.223, versus 0.218 for RB. This rejects that particular training remedy; it does not
prove that either representational family is incapable.

Finally, a convex weak-noise baseline used quadratic cumulative Clifford
toggling-frame features and lag correlations up to eight steps. Ridge strength and
memory lag were selected on validation data. At `idle180` bias 0.5 it produced
validation/test RMSE 0.21878/0.17822, worse than RB's 0.21771/0.17221. On `idle100`
it improved validation RMSE from 0.19936 to 0.17856 but failed to extrapolate, with
test RMSE 0.23184 versus RB's 0.18341. Thus this weak-noise feature approximation is
not a competitive long-horizon model.

## Scientific conclusion

The best supported statement is:

> For `idle100` biases 0.4–0.54, individual long-sequence survival probabilities are
> predicted by a compact, physically valid, time-homogeneous memoryless model as well
> as by a general memoryless channel and better than the previously tested OQE
> forecasts. The same active prediction advantage does not transfer to `idle180`.

This changes the next research target. Adding environmental dimension is not the
priority. The next model should learn a parsimonious correlated-noise path only after
the complete memoryless prediction has been subtracted, and it should be assessed on
interventions or compatible-process bounds capable of distinguishing retained
classical or quantum information.

## Reproduction

```bash
python scripts/run_markov_model_comparison.py data/external/pt_recovery \
  --condition idle100 --biases 0.4,0.5,0.52,0.54 \
  --models damped_d1,markov_cptp --seeds 0,1,2,3,4 --epochs 50 \
  --output results/markov_models_idle100_active.json

python scripts/run_markov_model_comparison.py data/external/pt_recovery \
  --condition idle180 --biases 0.1,0.4,0.5,0.52,0.54,0.6 \
  --models damped_d1,markov_cptp --seeds 0,1,2,3,4 --epochs 50 \
  --output results/markov_models_idle180_confirmation.json
```

Additional artifacts:

- `results/reconstructed_oqe_idle180_bias_05.json`
- `results/reconstructed_oqe_idle180_bias_05_scale_02.json`
- `results/reconstructed_oqe_idle180_bias_05_scale_05.json`
- `results/reconstructed_oqe_idle180_bias_05_scale_1.json`
- `results/reconstructed_oqe_idle180_bias_05_D4_D6.json`
- `results/memory_retain_idle180_screen.json`
- `results/quasistatic_gaussian_idle180_bias_05.json`
- `results/correlated_models_idle180_bias_05_centered2.json`
- `results/toggling_frame_idle100_bias_05.json`
- `results/toggling_frame_idle180_bias_05.json`
