#!/usr/bin/env python3
"""Exact native replay and paired batch-one service for frozen time templates."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from ptwm.native_time_templates import NativeTimeProgram, TimeProgramState, library
from ptwm.surface_program import LocalFeatures
from ptwm.time_templates import choose_action, difficulty_bins, mode_logits, mode_rate_filter, splice_noise
from scripts.benchmark_surface_frontier import quantiles, queue_stats, restore_model, restore_static
from scripts.run_surface_frontier_challenge import Seeds, circuit, make_decoder
from scripts.run_surface_time_templates import CONDITIONS, circuits, digest_data, generate


def balanced_order(names, replicate, repeat):
    names = list(names)
    if replicate % 2:
        names.reverse()
    offset = (replicate + repeat) % len(names)
    return names[offset:] + names[:offset]


def restore_time_static(artifact):
    name = artifact["selected"]["static"]
    if "_time_" in name:
        c = circuits()[int(name.rsplit("_", 1)[1])]
    elif name == "corr_optimized_time":
        rates = artifact["static_parameters"][name]["rates"]
        c = splice_noise(circuit(5, *rates[:2]), circuit(5, *rates[2:]), .5)
    else:
        return restore_static(artifact, name)
    return make_decoder(name, c.detector_error_model(decompose_errors=True), name.startswith("corr"), {})


def validate_native(artifact, *, records):
    mapper = LocalFeatures.from_circuit(circuits()[0])
    evidence = restore_model(artifact["evidence_model"])
    tables = {k: np.asarray(v) for k, v in artifact["risk_tables"].items()}
    rule = artifact["selected"]["four"].split(":")[-1]
    native = NativeTimeProgram(mapper, evidence, tables, rule=rule)
    checks, timing_data, timing_choices = {}, None, None
    horizon, streams = artifact["config"]["horizon"], artifact["config"]["streams"]
    for condition in CONDITIONS:
        d, y, modes = generate(Seeds(artifact["root_seed"], artifact["replicate"], 5),
                               "time_test", condition, streams, horizon)
        if digest_data(d, y, modes) != artifact["data_hashes"][f"time_test_{condition}"]:
            raise ValueError("simulator replay differs from frozen accuracy artifact")
        logits = mode_logits(mapper.transform(d), evidence).reshape(streams, horizon, 4)
        p, _ = mode_rate_filter(logits)
        expected = choose_action(p.reshape(-1, 4), tables, difficulty_bins(d), rule)
        d = np.ascontiguousarray(d, dtype=np.uint8)
        actual = np.empty(len(d), dtype=np.uint8)
        base, stride = d.ctypes.data, d.strides[0]
        for i in range(len(d)):
            if i % horizon == 0:
                native.reset()
            actual[i] = native.choose_address(base + i * stride)
        different = int(np.sum(actual != expected))
        if different:
            raise AssertionError(f"native choice differs on {different} records in {condition}")
        checks[condition] = {"records": len(d), "choice_disagreements": different}
        if condition == "four_way":
            timing_data, timing_choices = d[:records].copy(), expected[:records].copy()
    return native, timing_data, timing_choices, checks


def benchmark(path, *, records, repeats):
    artifact = json.loads(path.read_text())
    native, d, choices, checks = validate_native(artifact, records=records)
    static = restore_time_static(artifact)
    graphs = [make_decoder(str(i), c.detector_error_model(decompose_errors=True), True, {}) for i, c in enumerate(circuits())]
    decoded = np.column_stack([g.decode(d) for g in graphs])
    references = {"static": static.decode(d), "native_pipeline": decoded[np.arange(len(d)), choices],
                  "native_frontend": choices}

    def static_step(row, address):
        return static.matcher.decode(row, enable_correlations=static.correlated)[0]

    def pipeline_step(row, address):
        chosen = native.choose_address(address)
        return graphs[chosen].matcher.decode(row, enable_correlations=True)[0]

    def frontend_step(row, address):
        return native.choose_address(address)

    callables = {"static": static_step, "native_pipeline": pipeline_step, "native_frontend": frontend_step}
    rows = list(d)
    addresses = [d.ctypes.data + i * d.strides[0] for i in range(len(d))]
    for execute in callables.values():
        native.reset()
        for i in range(min(512, len(d))):
            execute(rows[i], addresses[i])
    samples, differences, orders = {k: [] for k in callables}, {k: 0 for k in callables}, []
    gc.disable()
    try:
        for repeat in range(repeats):
            order = balanced_order(callables, artifact["replicate"], repeat)
            orders.append(order)
            for name in order:
                elapsed, output = np.empty(len(d)), np.empty(len(d), dtype=np.uint8)
                execute = callables[name]
                for i, row in enumerate(rows):
                    if i % artifact["config"]["horizon"] == 0:
                        native.reset()
                    address = addresses[i]
                    start = time.perf_counter_ns()
                    output[i] = execute(row, address)
                    elapsed[i] = (time.perf_counter_ns() - start) / 1000
                samples[name].append(elapsed)
                differences[name] += int(np.sum(output != references[name]))
    finally:
        gc.enable()
    if any(differences.values()):
        raise AssertionError(f"native service changed output: {differences}")
    methods = {name: {"pooled": quantiles(np.concatenate(values)), "per_repeat": [quantiles(v) for v in values],
                      "queues_first_repeat": {str(cadence): queue_stats(values[0], cadence) for cadence in (5, 10, 20, 50)}}
               for name, values in samples.items()}
    return {"artifact": str(path), "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "replicate": artifact["replicate"], "policy": artifact["selected"]["four"],
            "static": artifact["selected"]["static"], "native_validation": checks,
            "records_per_repeat": len(d), "repeats": repeats, "method_orders": orders,
            "timed_output_disagreements": differences, "parameter_buffer_bytes": native.constant_bytes,
            "persistent_probability_bytes": native.persistent_bytes,
            "C_struct_bytes_including_probability_scratch_and_metadata": ctypes.sizeof(TimeProgramState),
            "methods": methods, "pipeline_to_static_p99_ratio": methods["native_pipeline"]["pooled"]["p99_us"] / methods["static"]["pooled"]["p99_us"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.records <= 0 or args.records % 512 or args.repeats <= 0:
        raise ValueError("require positive repeats and complete 512-shot streams")
    lib = library()
    source_paths = [__file__, "src/ptwm/native_time_templates.py", "src/ptwm/csrc/time_template_frontend.c",
                    "docs/surface-time-template-latency-protocol.md", "scripts/benchmark_surface_frontier.py",
                    "scripts/run_surface_time_templates.py", "src/ptwm/time_templates.py", "src/ptwm/surface_program.py",
                    "scripts/run_surface_frontier_challenge.py", "scripts/run_surface_regime_switch.py",
                    "scripts/run_surface_crossfit_compiler.py", "scripts/summarize_surface_frontier_challenge.py",
                    "src/ptwm/surface_mode.py", "src/ptwm/native_surface.py", "src/ptwm/csrc/surface_frontend.c"]
    output = {"protocol": "docs/surface-time-template-latency-protocol.md", "compiler": lib.compiler,
        "native_source_sha256": lib.source_sha256, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(Path(p).resolve().relative_to(Path.cwd())): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths},
        "platform": platform.platform(), "processor": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if platform.system() == "Darwin" else platform.processor(),
        "measurement_contract": "Warm batch-one service including native evidence/filter/action, Python/C crossing, matching and output assignment. "
        "Contiguous input already in RAM; excludes acquisition, allocation, per-stream reset and setup. GC disabled only while timing. "
        "CPU unpinned; unrelated host processes uncontrolled. Queue replay uses hypothetical periodic completed-record arrivals.", "records": []}
    for path, digest in output["source_sha256"].items():
        committed = subprocess.check_output(["git", "show", f"{output['code_commit']}:{path}"])
        if hashlib.sha256(committed).hexdigest() != digest:
            raise RuntimeError(f"timing source is uncommitted: {path}")
    for path in sorted(args.input_dir.glob("r*.json")):
        row = benchmark(path, records=args.records, repeats=args.repeats)
        output["records"].append(row)
        print(f"r{row['replicate']}: native p99={row['methods']['native_frontend']['pooled']['p99_us']:.3f}us; "
              f"pipeline/static p99={row['pipeline_to_static_p99_ratio']:.3f}; differences={row['timed_output_disagreements']}", flush=True)
        complete = (len(output["records"]) == 10 and {r["replicate"] for r in output["records"]} == set(range(10))
            and all(r["records_per_repeat"] == 32768 and r["repeats"] == 3
                    and set(r["native_validation"]) == set(CONDITIONS)
                    and all(c["records"] == 256 * 512 for c in r["native_validation"].values()) for r in output["records"]))
        output["gates"] = {"complete": complete, "service_cost": complete and all(r["pipeline_to_static_p99_ratio"] <= 1.25 for r in output["records"]),
            "all_choices_exact": all(c["choice_disagreements"] == 0 for r in output["records"] for c in r["native_validation"].values()),
            "timed_outputs_exact": all(not any(r["timed_output_disagreements"].values()) for r in output["records"])}
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p, digest in output["source_sha256"].items()):
        raise RuntimeError("timing source changed during run")


if __name__ == "__main__":
    main()
