#!/usr/bin/env python3
"""Benchmark a compiled affine-I/Q-to-edge-reweight hot-path kernel."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np

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
    # Seed a harmless maximum so every tested update remains on the Tier-1 path.
    dummy = matching.num_nodes
    matching.add_boundary_edge(
        dummy, fault_ids=set(), weight=float(np.log((1.0 - 1e-5) / 1e-5)),
        error_probability=1e-5,
    )
    detectors = np.pad(
        record["detectors"][test].astype(np.uint8), ((0, 0), (0, 1))
    )
    packed = pack_soft_reweighting(plan)
    args = (
        packed_head.weights, packed_head.feature_min, packed_head.feature_range,
        packed.endpoints, packed.residual_factor, packed.measurement_indices,
        packed.measurement_mask, packed.floor_probability, packed.ceiling_probability,
    )
    # Warm both Numba and matching before measuring.
    fused_affine_reweights_one(
        record["soft_measurements"][test[0]], record["hard_measurements"][test[0]], *args
    )
    matching.decode(detectors[0])

    composed_ns = np.empty(len(test))
    fused_ns = np.empty(len(test))
    composed_pipeline_ns = np.empty(len(test))
    fused_pipeline_ns = np.empty(len(test))
    maximum_weight_difference = 0.0
    disagreements = 0
    for position, row in enumerate(test):
        started = perf_counter_ns()
        probability = packed_head.predict(record["soft_measurements"][row])
        error = np.where(
            record["hard_measurements"][row], 1.0 - probability, probability
        )
        composed = packed.build(error)
        composed_ns[position] = perf_counter_ns() - started
        started = perf_counter_ns()
        fused = fused_affine_reweights_one(
            record["soft_measurements"][row], record["hard_measurements"][row], *args
        )
        fused_ns[position] = perf_counter_ns() - started
        maximum_weight_difference = max(
            maximum_weight_difference, float(np.max(np.abs(composed - fused)))
        )

        started = perf_counter_ns()
        first = matching.decode(detectors[position], edge_reweights=composed)[0]
        composed_pipeline_ns[position] = perf_counter_ns() - started + composed_ns[position]
        started = perf_counter_ns()
        second = matching.decode(detectors[position], edge_reweights=fused)[0]
        fused_pipeline_ns[position] = perf_counter_ns() - started + fused_ns[position]
        disagreements += int(first != second)

    commit = os.environ.get("PTWM_CODE_COMMIT")
    if not commit:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "data": {"file": path.name, "circuit_group": circuit_group,
                 "calibration_records": len(train), "benchmark_records": len(test)},
        "design": {
            "comparison": "composed NumPy packed front end versus fused Numba kernel",
            "matching_backend": "pinned mutable PyMatching fork, Tier-1 weights",
            "go": "zero prediction disagreements and fused p50 at least 2x faster",
        },
        "maximum_edge_array_difference": maximum_weight_difference,
        "prediction_disagreements": disagreements,
        "latency": {
            "composed_frontend": quantiles(composed_ns),
            "fused_frontend": quantiles(fused_ns),
            "composed_pipeline": quantiles(composed_pipeline_ns),
            "fused_pipeline": quantiles(fused_pipeline_ns),
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
