#!/usr/bin/env python3
"""Benchmark transactional C++ batch decoding over a preindexed weight plan."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np

FORK_URL = "https://github.com/stannum13/PyMatching"
FORK_COMMIT = "5e0e5472d082f204c2d1c88c51a2a3cca157144b"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.rigetti import (  # noqa: E402
    fit_measurement_iq_heads, fit_spitz_pairwise_matching, load_qec_record,
    measurement_error_signatures, pack_affine_iq_heads, pack_soft_reweighting,
    prepare_soft_reweighting, typed_circuit_noise_model,
)
from ptwm.rigetti_jit import fused_affine_reweights_one  # noqa: E402


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {name: float(np.quantile(values, q)) for name, q in (
        ("p50_ns", 0.5), ("p95_ns", 0.95), ("p99_ns", 0.99)
    )}


def run(
    path: Path, circuit_group: str, records: int, batch_sizes: list[int],
    syndrome_rounds: int,
) -> dict:
    import pymatching

    if "edge_reweights" not in inspect.signature(pymatching.Matching.decode_batch).parameters:
        raise RuntimeError("endpoint batch reweight API is required")
    if not hasattr(pymatching.Matching, "decode_batch_preindexed"):
        raise RuntimeError("batch-preindexed API is required")
    record = load_qec_record(str(path), circuit_group)
    boundary = int(0.6 * len(record["detectors"]))
    train = np.arange(boundary)
    test = np.arange(boundary, min(len(record["detectors"]), boundary + records))
    heads = fit_measurement_iq_heads(
        record["soft_measurements"], record["hard_measurements"],
        record["measurement_qubits"], train, knots=1, balanced=True,
    )
    packed_head = pack_affine_iq_heads(record["measurement_qubits"], heads)
    calibration_probability = packed_head.predict(record["soft_measurements"][train])
    calibration_error = np.where(
        record["hard_measurements"][train],
        1.0 - calibration_probability, calibration_probability,
    )
    template = pymatching.Matching.from_detector_error_model(typed_circuit_noise_model(
        record["circuit"], measurement_probability=0.03,
        one_qubit_probability=0.003, two_qubit_probability=0.03,
    ))
    matching, _ = fit_spitz_pairwise_matching(
        template, record["detectors"][train], floor_probability=1e-4
    )
    plan, _ = prepare_soft_reweighting(
        matching, measurement_error_signatures(record["circuit"]),
        np.mean(calibration_error, axis=0),
    )
    maximum_weight = float(np.log((1.0 - 1e-5) / 1e-5))
    dummy = matching.num_nodes
    matching.add_boundary_edge(
        dummy, fault_ids=set(), weight=maximum_weight, error_probability=1e-5
    )
    detectors = np.pad(
        record["detectors"][test].astype(np.uint8), ((0, 0), (0, 1))
    )
    packed = pack_soft_reweighting(plan)
    matching.compile_reweight_plan(np.asarray(packed.endpoints, dtype=np.int64))
    kernel_args = (
        packed_head.weights, packed_head.feature_min, packed_head.feature_range,
        packed.endpoints, packed.residual_factor, packed.measurement_indices,
        packed.measurement_mask, packed.floor_probability, packed.ceiling_probability,
    )
    updates = []
    for row in test:
        update = fused_affine_reweights_one(
            record["soft_measurements"][row], record["hard_measurements"][row],
            *kernel_args,
        )
        if np.any(update[:, 2] < 0) or np.any(update[:, 2] > maximum_weight):
            raise RuntimeError("weights escaped compiled Tier-1 normalization")
        updates.append(update)
    weight_matrix = np.ascontiguousarray(np.asarray(updates)[:, :, 2])
    baseline_before = matching.decode_batch(detectors[:100])
    curves = []
    for size in batch_sizes:
        count = (len(test) // size) * size
        endpoint_calls = []
        preindexed_calls = []
        prediction_disagreements = 0
        maximum_weight_difference = 0.0
        for block, start in enumerate(range(0, count, size)):
            stop = start + size
            endpoint_rules = updates[start:stop]
            if block % 2 == 0:
                started = perf_counter_ns()
                first, first_weights = matching.decode_batch(
                    detectors[start:stop], edge_reweights=endpoint_rules,
                    reweight_stride=1, return_weights=True,
                )
                endpoint_calls.append(perf_counter_ns() - started)
                started = perf_counter_ns()
                second, second_weights = matching.decode_batch_preindexed(
                    detectors[start:stop], weight_matrix[start:stop], return_weights=True
                )
                preindexed_calls.append(perf_counter_ns() - started)
            else:
                started = perf_counter_ns()
                second, second_weights = matching.decode_batch_preindexed(
                    detectors[start:stop], weight_matrix[start:stop], return_weights=True
                )
                preindexed_calls.append(perf_counter_ns() - started)
                started = perf_counter_ns()
                first, first_weights = matching.decode_batch(
                    detectors[start:stop], edge_reweights=endpoint_rules,
                    reweight_stride=1, return_weights=True,
                )
                endpoint_calls.append(perf_counter_ns() - started)
            prediction_disagreements += int(np.sum(first != second))
            maximum_weight_difference = max(
                maximum_weight_difference,
                float(np.max(np.abs(first_weights - second_weights))),
            )
        endpoint_calls = np.asarray(endpoint_calls, dtype=float)
        preindexed_calls = np.asarray(preindexed_calls, dtype=float)
        record_interval = syndrome_rounds * 1700.0
        fill = (size - 1) * record_interval
        curves.append({
            "batch_size": size,
            "records": count,
            "prediction_disagreements": prediction_disagreements,
            "maximum_solution_weight_difference": maximum_weight_difference,
            "endpoint_compute_batch": quantiles(endpoint_calls),
            "preindexed_compute_batch": quantiles(preindexed_calls),
            "endpoint_p50_ns_per_record": float(np.quantile(endpoint_calls / size, 0.5)),
            "preindexed_p50_ns_per_record": float(np.quantile(preindexed_calls / size, 0.5)),
            "completed_record_interval_ns": record_interval,
            "worst_case_fill_delay_ns": fill,
            "preindexed_oldest_record_p50_ns": fill + float(np.quantile(preindexed_calls, 0.5)),
        })
    baseline_after = matching.decode_batch(detectors[:100])
    if np.any(baseline_before != baseline_after):
        raise RuntimeError("batch calls did not restore the ordinary base graph")
    commit = os.environ.get("PTWM_CODE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "pymatching_module": str(Path(pymatching.__file__).resolve()),
            "fork_url": FORK_URL, "fork_commit": FORK_COMMIT,
            "install": f"python -m pip install git+{FORK_URL}.git@{FORK_COMMIT}",
        },
        "data": {
            "file": path.name, "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            "circuit_group": circuit_group, "calibration_records": len(train),
            "benchmark_records": len(test), "dynamic_edges": len(packed.endpoints),
        },
        "design": {
            "comparison": "endpoint batch API versus transactional C++ preindexed batch API",
            "order": "AB/BA by block; frontend excluded from timed calls",
            "go": "zero disagreements and >=20% per-record improvement or cadence crossing on both sessions",
        },
        "base_restore_disagreements": int(np.sum(baseline_before != baseline_after)),
        "curves": curves,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", required=True)
    parser.add_argument("--records", type=int, default=40_000)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--syndrome-rounds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.data, args.circuit_group, args.records, args.batch_sizes, args.syndrome_rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
