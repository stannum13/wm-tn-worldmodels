# Delay-aware streaming control: first directional screen

## Question

Does an explicit learned transition model improve control selected from a noisy
real-time record when the action takes effect later than the observation? This is the
smallest implementation of setting S2 in the redistributed plan. It represents a
classical detuning condition probed repeatedly, not a complete stochastic-master-
equation model of a continuously measured qubit.

## Design

The hidden condition is a two-state Markov process with transition probabilities
0.005 (normal to fault) and 0.04 (fault to normal) per sample. Probe emissions have
means -0.8 and 0.8 with unit Gaussian noise. Independently generated detector artifacts
occur with probability 0.005 and dropped samples with probability 0.002.

For each of 20 independent seeds, a two-state Gaussian HMM was fitted without hidden
state labels to 12 streams of length 1,200. It was evaluated on 40 fresh streams. The
comparison includes an instantaneous soft threshold, a robust EWMA, the fitted HMM's
current causal belief, and that belief propagated to the action time. All methods use
only samples available when the action is selected. The endpoint is squared error
between the soft compensation action and the hidden condition at actuation.

The run used commit `d16c27ea429125ebf7d70f6e6c0b671d47a6d398` on an 8-vCPU GCP
instance and took 2 minutes 26 seconds. The complete per-seed record is
[`results/streaming_benchmark.json`](../results/streaming_benchmark.json).

## Results

| Delay (samples) | Best non-forecast comparator | Comparator MSE | Forecast MSE | Paired reduction (95% t interval) |
|---:|---|---:|---:|---:|
| 0 | HMM current belief | 0.0245 | 0.0245 | 0.0% |
| 5 | HMM current belief | 0.0562 | 0.0523 | 6.9% (6.2–7.5%) |
| 10 | HMM current belief | 0.0815 | 0.0702 | 13.9% (12.9–14.9%) |
| 25 | HMM current belief | 0.1301 | 0.0931 | 28.4% (26.9–29.8%) |
| 50 | EWMA | 0.1625 | 0.1004 | 38.2% (36.9–39.5%) |
| 100 | EWMA | 0.1701 | 0.1013 | 40.5% (39.4–41.6%) |

The fitted transition means were 0.00548 and 0.0445, close to the generating values.
Forecasting is identical to filtering at zero delay and becomes useful as delay grows.
This supports the mechanism claim that propagating a causal belief to actuation time
can matter independently of improving the current-state estimator.

## Decision

The original analysis called this a screening GO at delays of 25 samples and longer.
An independent audit found that interpretation too strong. The score is Brier loss for
a future hidden state, not closed-loop control; the action does not alter the process.
Moreover, the stationary fault prior has theoretical Brier risk 0.0988 and is slightly
better than the fitted forecast at delays 50 and 100. The long-delay GO is withdrawn.
The remaining scientific target is the intermediate-delay region where observations
retain information beyond both a stationary prior and a robust direct predictor.

Before confirmation, freeze a comparator-selection rule and test fresh parameter
instances spanning dwell time, signal-to-noise ratio, artifact rate, and asymmetric
transition rates. Replace the binary artifact-response diagnostic with intervention
magnitude and incurred-cost metrics: at long delays the forecast often chooses a small
soft action, which the current thresholded diagnostic records as no action. Add a direct
history-aware policy and mismatched-model conditions. Only then extend the successful
regime to a backaction-consistent quantum trajectory and actual closed-loop fidelity.

## Parameter-family confirmation

A follow-up run froze the comparator using five development seeds for each of eight
new process settings, then evaluated ten separate seeds. The settings crossed two
signal amplitudes (0.4 and 0.8), two mean fault dwell times (about 17 and 50 samples),
and detector-artifact rates of 0.005 and 0.02. Delays were normalized to each fault
dwell time. The run used commit `8b66ff404a2510232f58365b4c9c69944d5d485f`
on the same GCP VM and consumed 2 minutes 38 seconds wall time across eight workers.

| Delay / mean fault dwell | Mean relative reduction across eight instances | 95% t interval across instances |
|---:|---:|---:|
| 0.10 | -5.0% | -26.7–16.7% |
| 0.25 | -0.5% | -18.0–17.0% |
| 0.50 | 7.2% | -7.2–21.6% |
| 1.00 | 18.8% | 13.2–24.4% |

The cross-cell intervals above are descriptive because the eight cells are fixed design
conditions, not random draws from a declared process population. The follow-up does
**not** confirm a general 20% benefit. One low-signal,
high-artifact instance was 52% worse at a quarter dwell time. At a full dwell time,
all instance means were positive but the cross-instance mean remained below the
minimum worthwhile effect. The first screen therefore located a favorable operating
point; it did not establish robust superiority.

The next mechanistic comparison should replace the Gaussian HMM emission with a
robust contamination model while keeping its transition forecast fixed. This directly
tests whether the observed failure comes from corrupted state estimation rather than
from the forecasting idea. The full confirmation record is
[`results/streaming_confirmation.json`](../results/streaming_confirmation.json).
