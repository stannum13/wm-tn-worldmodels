#!/usr/bin/env python3
"""Measure causal batch-one service, including features, filter and decoder calls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.native_surface import NativeProgram, library
from ptwm.surface_program import LocalFeatures
from scripts.run_surface_frontier_challenge import (
    REGIMES, Seeds, circuit, data_hash, decode_endpoints, family_predictions,
    make_decoder, pooled_dem, probability, stream,
)


def restore_model(model):
    return {k: np.asarray(v) if isinstance(v, list) else v for k, v in model.items()}


def restore_static(artifact, name):
    key = name.removeprefix("static:")
    parameters = artifact["static_parameters"][key]
    if "pooled_fraction" in parameters:
        dems = [circuit(artifact["distance"], *r).detector_error_model(decompose_errors=True) for r in REGIMES]
        dem = pooled_dem(*dems, parameters["pooled_fraction"])
    else:
        dem = circuit(artifact["distance"], parameters["gate"], parameters["measurement"]).detector_error_model(
            decompose_errors=True)
    return make_decoder(key, dem, key.startswith("corr"), parameters)


def quantiles(values):
    return {"mean_us": float(np.mean(values)), "p50_us": float(np.quantile(values, .5)),
            "p95_us": float(np.quantile(values, .95)), "p99_us": float(np.quantile(values, .99)),
            "p999_us": float(np.quantile(values, .999)), "max_us": float(np.max(values))}


def queue_stats(service, cadence):
    completion, responses = 0., []
    for index, elapsed in enumerate(service):
        arrival = index * cadence
        completion = max(arrival, completion) + elapsed
        responses.append(completion - arrival)
    return {**quantiles(responses), "ending_backlog_us": float(max(0, completion - len(service) * cadence))}


def benchmark(path, *, records, repeats):
    artifact = json.loads(path.read_text())
    distance, replicate = artifact["distance"], artifact["replicate"]
    horizon = artifact["config"]["horizon"]
    d, y, modes = stream(distance, seeds=Seeds(artifact["root_seed"], replicate, distance),
                         role="test", condition="nominal", streams=artifact["config"]["streams"], horizon=horizon)
    if data_hash(d, y, modes) != artifact["data_hashes"]["test_nominal"]:
        raise ValueError("replayed test differs from frozen artifact")
    d = np.ascontiguousarray(d[:records], dtype=np.uint8)
    records = len(d)
    mapper = LocalFeatures.from_circuit(circuit(distance, *REGIMES[0]))
    evidence = restore_model(artifact["models"]["evidence"])
    features = mapper.transform(d)
    programs, callables, references, native_states = {}, {}, {}, {}
    plain_static = restore_static(artifact, artifact["selected"]["static"])
    evidence_program = mapper.compile(evidence)
    endpoints = {backend: [make_decoder("endpoint", circuit(distance, *r).detector_error_model(
        decompose_errors=True), backend == "corr", {}) for r in REGIMES] for backend in ("mwpm", "corr")}
    decoded = {k: decode_endpoints(v, d) for k, v in endpoints.items()}
    probabilities = {k: probability(features, evidence, records // horizon, horizon, v)
                     for k, v in (("temporal", True), ("memoryless", False))}
    heads = {k: restore_model(v) for k, v in artifact["models"]["actions"].items()}
    expected = family_predictions(features, probabilities, decoded, heads, set(artifact["selected"].values()))

    def static_step(row, address):
        return plain_static.matcher.decode(row, enable_correlations=plain_static.correlated)[0]

    callables["static"] = static_step
    references["static"] = plain_static.decode(d)
    for label in ("one_decode", "compiled"):
        name = artifact["selected"][label]
        programs[label] = name
        if name.startswith("static:"):
            decoder = restore_static(artifact, name)
            callables[label] = lambda row, address, decoder=decoder: decoder.matcher.decode(
                row, enable_correlations=decoder.correlated)[0]
            references[label] = decoder.decode(d)
            continue
        backend, history, kind, value = name.split(":")
        native = NativeProgram(evidence_program, threshold=float(value) if kind == "threshold" else .5)
        if history == "memoryless":
            native.state.switch_probability = .5
        native_states[label] = native
        decoders = endpoints[backend]
        correlated = backend == "corr"
        references[label] = expected[name]
        if kind == "threshold":
            def step(row, address, native=native, decoders=decoders, correlated=correlated):
                chosen = native.choose_address(address)
                return decoders[chosen].matcher.decode(row, enable_correlations=correlated)[0]
        else:
            model = heads[name]
            raw_extra = model["weights"][-4:] / model["scale"][-4:]
            trimmed = {k: v[:-4] if isinstance(v, np.ndarray) else v for k, v in model.items()}
            trimmed["bias"] -= model["mean"][-4:] @ raw_extra
            action_native = NativeProgram(mapper.compile(trimmed))

            def step(row, address, native=native, decoders=decoders, correlated=correlated,
                     action_native=action_native, raw_extra=raw_extra):
                native.choose_address(address)
                p = min(max(native.state.probability, 1e-6), 1 - 1e-6)
                a, wa = decoders[0].matcher.decode(row, return_weight=True, enable_correlations=correlated)
                b, wb = decoders[1].matcher.decode(row, return_weight=True, enable_correlations=correlated)
                # Training deliberately used float32 energy features; preserve that contract.
                wa, wb = np.float32(wa), np.float32(wb)
                log_odds = np.float32(math.log(p) - math.log1p(-p))
                score = (action_native.score_address(address) + raw_extra[0] * log_odds
                         + raw_extra[1] * wa + raw_extra[2] * wb + raw_extra[3] * np.float32(wb - wa))
                return b[0] if score >= 0 else a[0]
        callables[label] = step

    samples, disagreements = {k: [] for k in callables}, {k: 0 for k in callables}
    addresses = d.ctypes.data + np.arange(records) * d.strides[0]
    rows = list(d)
    for label, execute in callables.items():
        if label in native_states:
            native_states[label].reset()
        for index in range(min(512, records)):
            execute(rows[index], int(addresses[index]))
    gc.disable()
    try:
        for repeat in range(repeats):
            order = np.random.default_rng(881 + repeat).permutation(list(callables))
            for label in order:
                elapsed, output = np.empty(records), np.empty(records, dtype=bool)
                execute = callables[label]
                for index, row in enumerate(rows):
                    if index % horizon == 0 and label in native_states:
                        native_states[label].reset()
                    address = int(addresses[index])
                    start = time.perf_counter_ns()
                    output[index] = execute(row, address)
                    elapsed[index] = (time.perf_counter_ns() - start) / 1000
                samples[label].append(elapsed)
                disagreements[label] += int(np.sum(output != references[label]))
    finally:
        gc.enable()
    if any(disagreements.values()):
        raise AssertionError(f"native service decisions changed: {disagreements}")
    summaries = {label: {"per_repeat": [quantiles(v) for v in arrays],
                          "pooled": quantiles(np.concatenate(arrays)),
                          "queues_first_repeat": {str(cadence): queue_stats(arrays[0], cadence)
                                                  for cadence in sorted({5, 10, 20, 50, distance})}}
                 for label, arrays in samples.items()}
    return {"artifact": str(path), "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "distance": distance, "replicate": replicate, "records_per_repeat": records,
            "repeats": repeats, "programs": programs, "decisions_differing_from_offline": disagreements,
            "methods": summaries,
            "one_decode_to_static_p99_ratio": summaries["one_decode"]["pooled"]["p99_us"] /
                                                  summaries["static"]["pooled"]["p99_us"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--records", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.records % 512:
        raise ValueError("use complete 512-shot streams for reset-equivalent profiling")
    output = {"platform": platform.platform(), "processor": subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip() if platform.system() == "Darwin"
        else platform.processor(), "compiler": library().compiler, "native_source_sha256": library().source_sha256,
        "measurement_contract": "Warm batch-one completed-shot service. Includes feature/filter, C/Python boundaries, "
        "and matching. Preexisting contiguous RAM input; excludes acquisition. GC disabled during timing; "
        "CPU not pinned on macOS. Queue numbers replay measured service under hypothetical periodic arrivals.",
        "artifacts": []}
    for path in args.artifacts:
        result = benchmark(path, records=args.records, repeats=args.repeats)
        output["artifacts"].append(result)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(f"{path.name}: p99 ratio {result['one_decode_to_static_p99_ratio']:.3f}, "
              f"decisions {result['decisions_differing_from_offline']}", flush=True)


if __name__ == "__main__":
    main()
