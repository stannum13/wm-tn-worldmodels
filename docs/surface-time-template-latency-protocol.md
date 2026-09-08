# Native time-template selector: service-cost protocol

Freeze implementation before timing, after the accuracy campaign finishes.
Do not change its selected policies. Profile all ten independently fitted raw R4
policies and their selected static comparators on the primary four-way condition.

The C kernel folds four affine feature maps into fixed detector/pair coefficients,
updates the same 20-probability state, and selects a graph using the frozen
mode/global/count-table rule. It does not decode or access logical truth. A single
Python/C call precedes one correlated PyMatching invocation. There are two internal
matching passes. No fast-math compiler flag or probability quantization is used.

Before timing, replay every condition/fit and require zero graph-choice differences
between native and frozen NumPy semantics. Validate every regenerated data hash.
Require zero final-decode disagreements on timed records as well.

Use 32,768 completed records and three repeats per method/fit. Warm 512 records,
reset mode state between independent 512-shot streams outside the timed section,
and counterbalance method order with a three-repeat Latin rotation. Rotate the
initial order by replicate modulo three and reverse it on odd replicates, so each
method occupies each run position exactly once per fit. Record every order.
Inputs already reside in
contiguous RAM. Include the native feature/filter/action call, Python/C crossing,
matching and output assignment. Exclude acquisition, preallocated storage, stream
reset, compilation and graph construction; publish these exclusions explicitly.
Disable GC only during timing, do not pin CPUs on macOS, and do not terminate or
control unrelated host processes. No accuracy campaign should run concurrently.

Report p50/p95/p99/p99.9, mean, maximum observed service time, all per-repeat
quantiles, constant bytes, probability-state bytes, and hypothetical FIFO responses
at 5/10/20/50-us completed-record arrival intervals. Graph storage is excluded from
controller byte counts; parameter-buffer totals also exclude compiled-in scalar
constants, executable code and object overhead. Verify all direct local source
hashes against the clean committed tree before profiling. Tail observations are
not hard worst-case guarantees.

Engineering GO: all ten pooled p99 native-pipeline/static ratios <=1.25 and every
semantic equivalence check passes. This is separate from every accuracy/transfer
gate, and cannot convert a failed transfer condition into a deployment GO.
