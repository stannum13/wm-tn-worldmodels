# Persistent schedule-evidence screen — 2026-09-09

Development-only follow-up, frozen before inspecting its results. The prior
per-record schedule estimator recovered only 7.1% of the selected-static to
supplied-schedule gap and rarely identified the exact schedule. This experiment
tests the specific explanation that a persistent physical schedule cannot be
reliably inferred from one sparse record but can be inferred across records.

Reuse the same stationary-only Bernoulli emissions, six detector-time bins, and
0/1/2-change schedule grammars. For each independent stream, update a score for
each candidate after every completed record:

    state[t] = retention * state[t-1] + record_log_likelihood[t]

and select the highest score after subtracting the previously defined change
penalty. State is reset between streams and conditions. The current record is
included before its graph is selected; this is completed-record decoding, not
within-cycle feedback. Compare retention {0,.5,.9,.99,1}; retention 0 reproduces
memoryless per-record inference. Select retention and change penalty jointly on
separate selection streams for each schedule-width family.

Distance 5, root seed 2026090914, three replicates. Per replicate use fresh
65,536-shot stationary calibration per mode, then for each of the same 14 physical
schedules use 16 selection streams x 128 records and 64 development streams x
256 records. Each physical schedule remains fixed within a stream. Report complete
and post-burn-in LER at burn-ins 0,4,16,64, exact schedule rate over time, state
bytes, rescues/harms, action equivalence, and all seeds/data/source hashes.

This favorable persistent-condition screen is a mechanism test. It is not evidence
of tracking when schedules switch between records. Proceed to a switching test
only if memory improves both logical error and identification. No confirmatory
gate or SOTA claim is evaluated here.
