# Native graph selector: exact replay, low frontend cost, failed tail gate

8 September 2026. The [latency protocol](surface-time-template-latency-protocol.md)
and implementation were frozen at `0267b56`. This profiles the ten policies from
the [time-template accuracy experiment](surface-time-template-report.md), without
changing their selection or treating timing replay as fresh accuracy data.
Raw measurements: [surface_time_template_latency.json](../results/surface_time_template_latency.json).

## Verdict

**Semantic GO; engineering tail-cost NO-GO; deployment NO-GO.** Native graph choices
agree on all 10,485,760 replayed records across eight conditions and ten fits.
The timed native pipeline also has zero output differences on 983,040 repeated
record decisions. Nine of ten p99 pipeline/static ratios meet the <=1.25 limit,
but fit 4 reaches 1.5434. That failure is retained, not discarded as an outlier.

| Measurement across ten fits | Minimum | Maximum |
| --- | ---: | ---: |
| Native frontend mean | 1.213 µs | 1.399 µs |
| Native frontend p99 | 1.583 µs | 2.042 µs |
| Full native pipeline mean | 12.758 µs | 15.952 µs |
| Full native pipeline p50 | 11.500 µs | 12.083 µs |
| Full native pipeline p99 | 31.083 µs | 58.454 µs |
| Selected static pipeline p99 | 30.457 µs | 37.874 µs |
| Per-fit pooled p99 ratio | 0.914 | 1.543 |

The minimum ratio is not a general speedup claim: costs vary with graph, syndrome
trajectory and host conditions. Median cost is higher in all ten fits; mean cost
is higher in nine of ten.

## What was compiled

Four affine local-feature models are folded into fixed detector and local-pair
coefficients. A C kernel evaluates the evidence, updates the same joint distribution
over five rate hypotheses and four modes, and applies the frozen mode/global-risk/
count-risk rule. One Python/C crossing precedes one selected correlated-matching
invocation, with two internal matching passes.

Every fitted program uses 13,344 bytes of parameter buffers and 160 bytes of
persistent probabilities. The C struct occupies 288 bytes including those
probabilities, scratch arrays and pointer/dimension metadata. Parameter-buffer
counts exclude compiled-in scalar constants, executable code, Python objects,
input/output buffers, and the four decoder graphs. They are not total deployment
memory or an FPGA area measurement.

Sources: [native wrapper](../src/ptwm/native_time_templates.py),
[C kernel](../src/ptwm/csrc/time_template_frontend.c),
[benchmark](../scripts/benchmark_surface_time_templates.py).

## Why the failed ratio cannot be waved away

For fit 4, the three pipeline p99 observations are 57.736, 33.375 and 92.791 µs;
the corresponding static observations are 44.806, 37.598 and 34.250 µs.
This substantial variability warrants controlled replication and system
instrumentation. It does not establish a particular cause. The other nine pooled
ratios lie between 0.914 and 1.047, but selecting only those nine would violate
the frozen condition.

Maximum observed pipeline service reaches 22.181 ms; even frontend-only maximum
service reaches 2.255 ms. These measurements include scheduling effects and are
not deterministic upper bounds on either the kernel or system. The static path
also has millisecond maxima. No host-level hard-real-time guarantee follows from
the small arithmetic kernel.

## Queueing is a separate failure

At hypothetical 5-µs completed-record arrivals, replaying each fit's first measured
pipeline service trace produces 251–367 ms of ending backlog after 32,768 records.
Every mean service time exceeds that arrival interval. A faster controller alone
does not close the decoder-throughput gap. The raw artifact also reports response
quantiles/backlog at 10, 20 and 50 µs; these are hypothetical FIFO replays, not
measured streaming hardware response times.

## Measurement contract and review

Apple M2 Max; Apple clang 17.0.0; `-O3` without fast-math. Three repeats of 32,768
completed records per method and fit, after a 512-record warmup. Balanced Latin
rotations ensure each method occurs first, second and third once per fit. The
initial proposed pseudorandom seeds accidentally produced identical orders; an
independent review caught and corrected that before the frozen measurement.

Inputs are already contiguous in RAM. Timings include native evidence/filter/
action, Python/C crossing, matching and output assignment, but exclude acquisition,
preallocation, graph construction, compilation and between-stream state reset.
GC is disabled only while timing. CPUs are unpinned and unrelated host processes
are uncontrolled; no accuracy campaign ran concurrently. Fifteen local source
hashes were verified against the committed tree before execution and checked for
changes afterward. All replay dataset hashes agree with the accuracy artifacts.

This establishes a reproducible native implementation of the bounded adaptation
law. It does not rescue the separate unseen-cutpoint failure, establish universal
tail latency, or outperform published real-time neural decoders.
