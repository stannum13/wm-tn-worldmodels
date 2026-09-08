#!/usr/bin/env python3
"""Compare endpoint-parsed and graph-preindexed mutable matching hot paths."""

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

BASE_FORK_URL = "https://github.com/Allenator/PyMatching"
BASE_FORK_COMMIT = "435dc7ec85c10314c09f069a3d924d3a3dee8251"
PREINDEXED_FORK_URL = "https://github.com/stannum13/PyMatching"
PREINDEXED_FORK_COMMIT = "f53805b6acbadadc13dc314c9791912c99313747"
PREINDEXED_PATCH_SHA256 = "d5bc0de2dad9c1c3c6c4bbeaa02c7cf9a0c8fc79f161be7432657b6de362c7fe"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.rigetti import (  # noqa: E402
    fit_measurement_iq_heads,
    fit_spitz_pairwise_matching,
    load_qec_record,
    measurement_error_signatures,
    pack_affine_iq_heads,
    pack_soft_reweighting,
    prepare_soft_reweighting,
    typed_circuit_noise_model,
)
from ptwm.rigetti_jit import fused_affine_reweights_one  # noqa: E402


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {name: float(np.quantile(values, q)) for name, q in (
        ("p50_ns", 0.5), ("p95_ns", 0.95), ("p99_ns", 0.99)
    )}


def run(path: Path, circuit_group: str, records: int) -> dict:
    import pymatching

    if "edge_reweights" not in inspect.signature(pymatching.Matching.decode).parameters:
        raise RuntimeError("endpoint edge_reweights API is required")
    if not all(hasattr(pymatching.Matching, name) for name in (
        "compile_reweight_plan", "decode_preindexed"
    )):
        raise RuntimeError("experimental preindexed API is required")

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
    endpoints = np.asarray(packed.endpoints, dtype=np.int64, order="C")
    matching.compile_reweight_plan(endpoints)
    kernel_args = (
        packed_head.weights, packed_head.feature_min, packed_head.feature_range,
        packed.endpoints, packed.residual_factor, packed.measurement_indices,
        packed.measurement_mask, packed.floor_probability, packed.ceiling_probability,
    )
    fused_affine_reweights_one(
        record["soft_measurements"][test[0]], record["hard_measurements"][test[0]],
        *kernel_args,
    )
    baseline_before = np.asarray([matching.decode(row)[0] for row in detectors[:100]])

    updates = []
    for row in test:
        update = fused_affine_reweights_one(
            record["soft_measurements"][row], record["hard_measurements"][row],
            *kernel_args,
        )
        if np.any(update[:, 2] < 0) or np.any(update[:, 2] > maximum_weight):
            raise RuntimeError("weights escaped compiled Tier-1 normalization")
        updates.append(update)

    endpoint_ns = np.empty(len(test))
    preindexed_ns = np.empty(len(test))
    endpoint_pipeline_ns = np.empty(len(test))
    preindexed_pipeline_ns = np.empty(len(test))
    disagreements = 0
    weight_differences = np.empty(len(test))
    for position in range(len(test)):
        endpoint_update = updates[position]
        indexed_weights = np.ascontiguousarray(endpoint_update[:, 2])
        if position % 2 == 0:
            started = perf_counter_ns()
            first, first_weight = matching.decode(
                detectors[position], edge_reweights=endpoint_update, return_weight=True
            )
            endpoint_ns[position] = perf_counter_ns() - started
            started = perf_counter_ns()
            second, second_weight = matching.decode_preindexed(
                detectors[position], indexed_weights, return_weight=True
            )
            preindexed_ns[position] = perf_counter_ns() - started
        else:
            started = perf_counter_ns()
            second, second_weight = matching.decode_preindexed(
                detectors[position], indexed_weights, return_weight=True
            )
            preindexed_ns[position] = perf_counter_ns() - started
            started = perf_counter_ns()
            first, first_weight = matching.decode(
                detectors[position], edge_reweights=endpoint_update, return_weight=True
            )
            endpoint_ns[position] = perf_counter_ns() - started
        disagreements += int(np.any(first != second))
        weight_differences[position] = abs(first_weight - second_weight)

    # Direct contiguous pipelines in a separate AB/BA-balanced pass.
    for position, row in enumerate(test):
        if position % 2 == 0:
            started = perf_counter_ns()
            endpoint_update = fused_affine_reweights_one(
                record["soft_measurements"][row], record["hard_measurements"][row],
                *kernel_args,
            )
            matching.decode(detectors[position], edge_reweights=endpoint_update)
            endpoint_pipeline_ns[position] = perf_counter_ns() - started
            started = perf_counter_ns()
            indexed_update = fused_affine_reweights_one(
                record["soft_measurements"][row], record["hard_measurements"][row],
                *kernel_args,
            )
            matching.decode_preindexed(detectors[position], indexed_update[:, 2])
            preindexed_pipeline_ns[position] = perf_counter_ns() - started
        else:
            started = perf_counter_ns()
            indexed_update = fused_affine_reweights_one(
                record["soft_measurements"][row], record["hard_measurements"][row],
                *kernel_args,
            )
            matching.decode_preindexed(detectors[position], indexed_update[:, 2])
            preindexed_pipeline_ns[position] = perf_counter_ns() - started
            started = perf_counter_ns()
            endpoint_update = fused_affine_reweights_one(
                record["soft_measurements"][row], record["hard_measurements"][row],
                *kernel_args,
            )
            matching.decode(detectors[position], edge_reweights=endpoint_update)
            endpoint_pipeline_ns[position] = perf_counter_ns() - started

    baseline_after = np.asarray([matching.decode(row)[0] for row in detectors[:100]])
    if np.any(baseline_before != baseline_after):
        raise RuntimeError("preindexed calls did not restore the base graph")
    commit = os.environ.get("PTWM_CODE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pymatching_module": str(Path(pymatching.__file__).resolve()),
            "base_fork_url": BASE_FORK_URL,
            "base_fork_commit": BASE_FORK_COMMIT,
            "preindexed_fork_url": PREINDEXED_FORK_URL,
            "preindexed_fork_commit": PREINDEXED_FORK_COMMIT,
            "preindexed_patch_sha256": PREINDEXED_PATCH_SHA256,
            "preindexed_install": (
                f"python -m pip install git+{PREINDEXED_FORK_URL}.git@"
                f"{PREINDEXED_FORK_COMMIT}"
            ),
        },
        "data": {
            "file": path.name, "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            "circuit_group": circuit_group, "calibration_records": len(train),
            "benchmark_records": len(test), "dynamic_edges": len(endpoints),
        },
        "design": {
            "comparison": "endpoint-parsed Tier-1 API versus graph-owned preindexed Tier-1 API",
            "order": "separate matching and contiguous-pipeline passes; AB/BA by record",
            "go": "zero prediction/weight disagreements and >=20% lower matching p50",
        },
        "prediction_disagreements": disagreements,
        "maximum_decode_weight_difference": float(np.max(weight_differences)),
        "base_restore_disagreements": int(np.sum(baseline_before != baseline_after)),
        "latency": {
            "endpoint_matching": quantiles(endpoint_ns),
            "preindexed_matching": quantiles(preindexed_ns),
            "endpoint_fused_pipeline": quantiles(endpoint_pipeline_ns),
            "preindexed_fused_pipeline": quantiles(preindexed_pipeline_ns),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", required=True)
    parser.add_argument("--records", type=int, default=2_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.data, args.circuit_group, args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
