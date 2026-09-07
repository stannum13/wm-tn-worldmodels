#!/usr/bin/env python3
"""Benchmark a compiled affine-I/Q-to-edge-reweight hot-path kernel."""

from __future__ import annotations

import argparse
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

MUTABLE_BACKEND_URL = "https://github.com/Allenator/PyMatching"
MUTABLE_BACKEND_COMMIT = "435dc7ec85c10314c09f069a3d924d3a3dee8251"


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {name: float(np.quantile(values, q)) for name, q in (
        ("p50_ns", 0.5), ("p95_ns", 0.95), ("p99_ns", 0.99)
    )}


def run(path: Path, circuit_group: str, records: int) -> dict:
    import pymatching

    if "edge_reweights" not in inspect.signature(pymatching.Matching.decode).parameters:
        raise RuntimeError("pinned mutable PyMatching edge_reweights API is required")

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
    maximum_weight = float(np.log((1.0 - 1e-5) / 1e-5))
    matching.add_boundary_edge(
        dummy, fault_ids=set(), weight=maximum_weight,
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
    composed_matching_ns = np.empty(len(test))
    fused_matching_ns = np.empty(len(test))
    maximum_weight_difference = 0.0
    disagreements = 0
    def composed_update(row: int) -> np.ndarray:
        probability = packed_head.predict(record["soft_measurements"][row])
        error = np.where(
            record["hard_measurements"][row], 1.0 - probability, probability
        )
        return packed.build(error)

    def fused_update(row: int) -> np.ndarray:
        return fused_affine_reweights_one(
            record["soft_measurements"][row], record["hard_measurements"][row], *args
        )

    for position, row in enumerate(test):
        started = perf_counter_ns()
        composed = composed_update(row)
        composed_ns[position] = perf_counter_ns() - started
        started = perf_counter_ns()
        fused = fused_update(row)
        fused_ns[position] = perf_counter_ns() - started
        maximum_weight_difference = max(
            maximum_weight_difference, float(np.max(np.abs(composed - fused)))
        )
        if np.any(fused[:, 2] < 0) or np.any(fused[:, 2] > maximum_weight):
            raise RuntimeError("fused weights escaped the seeded Tier-1 range")

        # Time identical matching inputs in alternating order, separately from
        # the direct end-to-end pipeline timings below.
        if position % 2 == 0:
            started = perf_counter_ns()
            matching.decode(detectors[position], edge_reweights=composed)
            composed_matching_ns[position] = perf_counter_ns() - started
            started = perf_counter_ns()
            matching.decode(detectors[position], edge_reweights=fused)
            fused_matching_ns[position] = perf_counter_ns() - started
        else:
            started = perf_counter_ns()
            matching.decode(detectors[position], edge_reweights=fused)
            fused_matching_ns[position] = perf_counter_ns() - started
            started = perf_counter_ns()
            matching.decode(detectors[position], edge_reweights=composed)
            composed_matching_ns[position] = perf_counter_ns() - started

        # Alternate order to avoid consistently giving either path the second-decode
        # cache/thermal position. Pipeline timings include fresh front-end execution.
        if position % 2 == 0:
            started = perf_counter_ns()
            first = matching.decode(
                detectors[position], edge_reweights=composed_update(row)
            )[0]
            composed_pipeline_ns[position] = perf_counter_ns() - started
            started = perf_counter_ns()
            second = matching.decode(
                detectors[position], edge_reweights=fused_update(row)
            )[0]
            fused_pipeline_ns[position] = perf_counter_ns() - started
        else:
            started = perf_counter_ns()
            second = matching.decode(
                detectors[position], edge_reweights=fused_update(row)
            )[0]
            fused_pipeline_ns[position] = perf_counter_ns() - started
            started = perf_counter_ns()
            first = matching.decode(
                detectors[position], edge_reweights=composed_update(row)
            )[0]
            composed_pipeline_ns[position] = perf_counter_ns() - started
        disagreements += int(first != second)

    commit = os.environ.get("PTWM_CODE_COMMIT")
    if not commit:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pymatching_module": str(Path(pymatching.__file__).resolve()),
            "pymatching_fork_url": MUTABLE_BACKEND_URL,
            "pymatching_fork_commit": MUTABLE_BACKEND_COMMIT,
            "pymatching_install": (
                f"python -m pip install git+{MUTABLE_BACKEND_URL}.git@"
                f"{MUTABLE_BACKEND_COMMIT}"
            ),
        },
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
            "composed_matching_call": quantiles(composed_matching_ns),
            "fused_matching_call": quantiles(fused_matching_ns),
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
