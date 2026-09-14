# First directional screens — 2026-09-09

These are local development runs, not SOTA or confirmatory results. The protocol
was written before these outcomes were inspected. Existing archived campaigns
were not changed. See `directional-feature-screen-protocol.md` for scope and
`neurosymbolic-architecture-spec-2026-09-09.md` for the proposed subsequent search.

## 1. Feature sufficiency: naive additions did not help in this screen

Three frozen-evidence replicates; each has 32,768 fitting, 32,768 selection, and
32,768 development-evaluation records. Thus 294,912 unique records were generated,
98,304 for development evaluation. The actions are the same two endpoint decoders
for every head. Existing calibration/static artifacts are reused, so these are
not three fresh full-pipeline refits. The dictionary contains 80 connected triple
and 176 connected four-detector parity motifs. The original action dictionary has
67 columns; all capped variants select at most 64 columns using fitting outcomes.

| Head | Mean development LER | Exact-model expected risk on the evaluated histories |
| --- | ---: | ---: |
| Uncapped original features, outcome fitting | 1.9826% | 1.9578% |
| Uncapped original features, teacher fitting | 1.9450% | 1.9272% |
| Capped base features, outcome fitting | 1.9765% | 1.9552% |
| Capped base features, teacher fitting | 1.9399% | 1.9219% |
| Parity dictionary, teacher fitting | 1.9887% | 1.9521% |
| Belief-interaction dictionary, teacher fitting | 1.9714% | 1.9523% |
| Both dictionaries, teacher fitting | 1.9989% | 1.9467% |
| Frozen static anchor | 1.9887% | 1.9986% |
| Restricted exact teacher | 1.8229% | 1.7875% |
| Full exact teacher | 1.7782% | 1.7409% |

The capped base teacher head has the lowest mean among these student arms. Its
point recovery against the uncapped outcome anchor is 26.75%, well short of the
previous 80% target. This is a descriptive ratio from a different small protocol,
not an improvement of the archived 11.3% confirmatory result.

The parity-teacher minus base-teacher differences are positive in all three fits;
the unadjusted, three-replicate Student-t descriptive interval is
[+0.0357,+0.0620] percentage points. The combined-dictionary difference is
+0.0590 pp with interval [-0.0758,+0.1938] pp. These post-screen descriptive
intervals are not multiplicity-adjusted claims, and three fits do not settle the
scientific hypothesis. Training contained only 648,690,731 endpoint disagreements.

Interpretation: enlarging a dictionary and ranking individual columns by marginal
outcome correlation was not enough. This screen does not falsify higher-order
features, sheaf models, or nonlinear interactions in general. Column selection
can discard jointly useful features and the fitting sample is small. It does
argue against escalating this particular dictionary/selector directly into an
expensive architecture campaign without a better diagnostic.

No sheaf relation or graph-local recurrent state was trained. The existing global
mode belief was used in the interaction features. Teacher probabilities remain
offline supervision/evaluation; no privileged state enters student features.
Both decoder energies are inputs: no one-decode or hot-path latency claim follows.

Run artifact: `results/directional_feature_screen_20260909.json`. It records
per-stream errors, rescues/harms, expected risks, selected columns, fitted models,
seed/data/source hashes, and source calibration artifacts. The initial attempt
completed replicate 0 but failed while serializing NumPy integers in motif tuples.
Only serialization typing was corrected; a regression test was added and the
same seeds and unchanged modeling protocol were rerun. Replicate-0 printed
results were reproduced. No failed or successful test partition was retuned.

## 2. Temporal composition: known schedules expose a real opportunity

The new compiler takes aligned circuits plus one mode per tick slot, including
the final readout slot. It preserves the complete ideal circuit and delegates
cross-seam fault propagation to Stim after composition. It does not concatenate
independently generated detector error models.

All 14 d3/d5 AB/BA one-change and ABA two-change cases reproduced the existing
splice reference circuit exactly, with identical detector coordinates. Each case
used 32,768 fresh shots: 458,752 total. Selected d5 outcomes are:

| Schedule | Known-schedule graph | Midpoint graph | Stationary A | Stationary B |
| --- | ---: | ---: | ---: | ---: |
| AB, one-third cut | 2.0416% | 2.2583% | 2.6581% | 2.5665% |
| AB, two-thirds cut | 1.1475% | 1.3214% | 1.2878% | 2.1362% |
| BA, two-thirds cut | 1.5137% | 1.6937% | 2.0325% | 2.0386% |
| ABA, cuts at thirds | 1.2878% | 1.9196% | 1.4801% | 2.2705% |

On the final row, the midpoint graph makes 629 errors versus 422 with the known
schedule. The change rescues 301 records and harms 94: a net 207 fewer errors,
not 301 net rescues. Relative to stationary A, it rescues 148 and harms 85.
This is a selected illustrative condition, not a pooled benchmark advantage.

Not every case improves: at d3 AB one-third, the known-schedule graph gives
2.4567% versus midpoint 2.4078%. A true-noise matching graph is not a Bayes-optimal
logical decoder, and sampling variation also matters. The complete negative and
positive outcomes are retained in `results/noise_schedule_screen_20260909.json`.

Crucially, the schedule is supplied. No inferred-cut controller was tested, no
external SOTA baseline was beaten, and this is not the previously proposed 70%
transfer-gain gate. These comparisons also do not reproduce the stronger tuned
static-candidate search from earlier campaigns. The result supports making valid
time structure representable before fitting a schedule estimator.

## 3. Inferred schedule: representation is no longer the main bottleneck

The bounded schedule estimator used stationary-only per-detector Bernoulli
emissions and enumerated 2/12/32 sequences with at most zero/one/two changes over
six detector-time bins. Three replicates used 131,072 stationary calibration,
57,344 selection, and 229,376 development-evaluation shots each. Across the
688,128 development-evaluation shots, equal-weighting all 14 conditions gives:

| Method | Aggregate LER | Supplied-schedule gap recovery |
| --- | ---: | ---: |
| Selected fixed compiled graph | 1.7609% | 0% |
| Inferred zero-change family | 1.7504% | 2.7% |
| Inferred one-change family | **1.7337%** | **7.1%** |
| Inferred two-change family | 1.7357% | 6.5% |
| Supplied physical schedule | 1.3768% | 100% reference |

The one-change minus static paired differences by replicate are -0.0323,
-0.0344 and -0.0148 percentage points. These are descriptive development results,
not a confirmatory interval. Exact one-change sequence identification is typically
only 3–8% per representable mixed condition. Exact ABA/BAB identification by the
two-change estimator is below 0.8% in every replicate. All routed predictions
match their predecoded candidate choices exactly.

This falsifies the useful version of the **per-record independent stationary
marginal** estimator: adding schedule width does not recover much of the available
logical improvement. It does not falsify schedule inference generally. Detector
events are sparse, boundary emissions differ from stationary emissions, and the
experiment discarded evidence after every record even though drift modes are
normally persistent across records. The next diagnostic is therefore a bounded
hypothesis filter over records, not more within-record graph templates.

Run artifact: `results/inferred_schedule_screen_20260909.json`.

## 4. Persistent memory changes the result

The prespecified follow-up kept each physical schedule fixed within a 256-record
stream and accumulated bounded candidate scores across completed records. Three
replicates and the same 14 conditions produced 688,128 development-evaluation
records. Hyperparameters were selected on separate streams. Equal-weighting the
conditions gives:

| Method | State (float64) | Aggregate LER | Supplied-schedule gap recovery |
| --- | ---: | ---: | ---: |
| Selected fixed compiled graph | 0 B | 1.7831% | 0% |
| Persistent zero-change family | 16 B | 1.6931% | 21.5% |
| Persistent one-change family | 96 B | 1.4579% | 77.8% |
| Persistent two-change family | 256 B | **1.4367%** | **82.9%** |
| Supplied physical schedule | privileged | 1.3650% | 100% reference |

The two-change controller reduces LER by 19.4% relative to the selected static
graph. Its static-minus-adaptive differences are +0.3313, +0.3932 and +0.3148
percentage points in the three replicates. Across all paired records it rescues
5,462 static failures and introduces 3,078 harms, a net 2,384 fewer errors. The
one-change family has 5,408 rescues and 3,170 harms. All selected graph indices
reproduce the separately predecoded candidate outputs exactly.

The mechanism is visible in identification, not just LER. For an AB half-cut,
the one-change family identifies the exact schedule on 8.3% of streams after one
record, 40.6% after 16, 62.5% after 64 and 83.3% after 256. The corresponding BA
figures end at 84.4%. Both stationary modes reach 100% by record 128. Exact
two-change identification remains harder: ABA/BAB thirds reach only 38.5%/40.6%
at record 256, although the two-change family improves their mean LER relative to
the one-change family from 1.5035%/1.7212% to 1.4364%/1.4404%.

This is a strong positive mechanism result for **persistent structural belief**,
not for generic recurrence. The zero-change family remains weak and sometimes
harms, whereas capacity matched to the physical schedule closes most of the
privileged reference gap. The selected retentions are almost always 0.99 or 1.0;
one replicate selected memoryless zero-change inference, which is further evidence
that persistence only helps when the hypothesis grammar can express the defect.

The favorable setup is also the main limitation: schedules do not switch between
records, all candidates are compiled in advance, float64 storage is reported
rather than synthesized hardware cost, and the same simulator family supplies
calibration and evaluation. This is development evidence, not SOTA, deployment,
or a tracking claim. Source hashes match the completed artifact.

Run artifact: `results/schedule_memory_screen_20260909.json`. Frozen protocol:
`docs/schedule-memory-screen-protocol.md`.

## 5. Next implementation decision

The decisive between-record switching experiment has now run with a matched
change detector/reset rule and dwell-time sweeps across the observed evidence
timescale. The 70% gate failed: the 256-byte controller recovered 37.1% at nominal
dwell 64. The complete phase boundary and resulting tracking diagnosis are in
`docs/switching-schedule-phase-diagram-report.md`.

For feature learning, diagnose the decision gap using conditional expected risk
and compare joint feature selection or boundary logical summaries before paying
for a large sheaf/neural architecture search. Keep the action set and observation
timing matched within each comparison.

## Reproduction

The commands below are relative to the repository and assume the existing
environment with NumPy, SciPy, Stim and PyMatching. Use a new output path; runners
refuse to overwrite previous results.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/wmtn-qec-venv/bin/python scripts/run_directional_feature_screen.py --output results/directional_feature_screen_replay.json
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/wmtn-qec-venv/bin/python scripts/run_noise_schedule_screen.py --output results/noise_schedule_screen_replay.json
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/wmtn-qec-venv/bin/python scripts/run_inferred_schedule_screen.py --output results/inferred_schedule_screen_replay.json
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/wmtn-qec-venv/bin/python scripts/run_schedule_memory_screen.py --output results/schedule_memory_screen_replay.json
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/wmtn-qec-venv/bin/python -m pytest -q
```

The full suite reports **151 passed, 1 skipped, 2 warnings**. The optional Numba
test is skipped because that dependency is not installed; two existing Torch
warnings remain. No cloud resources were launched.
