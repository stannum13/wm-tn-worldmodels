#!/usr/bin/env python3
"""Measure logical-error sensitivity to quantized per-measurement soft information."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

FORK_URL = "https://github.com/stannum13/PyMatching"
FORK_COMMIT = "edf4ea3"

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
from ptwm.rigetti_jit import fused_affine_quantized_weights_one  # noqa: E402


def interval(values: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def run(
    path: Path, circuit_group: str, records: int, bits: list[int], block_size: int
) -> dict:
    import pymatching

    if len(bits) != len(set(bits)):
        raise ValueError("probability bit widths must be unique")
    if not bits or any(bit < 0 or bit > 16 for bit in bits):
        raise ValueError("probability bit widths must be unique integers in [0, 16]")

    record = load_qec_record(str(path), circuit_group)
    boundary = int(0.6 * len(record["detectors"]))
    train = np.arange(boundary)
    test = np.arange(boundary, min(len(record["detectors"]), boundary + records))
    if len(test) % block_size:
        raise ValueError("test record count must be divisible by block size")
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
    matching, graph_diagnostics = fit_spitz_pairwise_matching(
        template, record["detectors"][train], floor_probability=1e-4
    )
    plan, soft_diagnostics = prepare_soft_reweighting(
        matching, measurement_error_signatures(record["circuit"]),
        np.mean(calibration_error, axis=0),
    )
    maximum_weight = float(np.log((1.0 - 1e-5) / 1e-5))
    matching.add_boundary_edge(
        matching.num_nodes, fault_ids=set(), weight=maximum_weight,
        error_probability=1e-5,
    )
    packed = pack_soft_reweighting(plan)
    matching.compile_reweight_plan(np.asarray(packed.endpoints, dtype=np.int64))
    detectors = np.pad(
        record["detectors"][test].astype(np.uint8), ((0, 0), (0, 1))
    )
    labels = record["observables"][test, 0].astype(np.uint8)
    kernel_args = (
        packed_head.weights, packed_head.feature_min, packed_head.feature_range,
        packed.residual_factor, packed.measurement_indices, packed.measurement_mask,
        packed.floor_probability, packed.ceiling_probability,
    )
    # Compile the Numba specialization outside measured scientific output.
    fused_affine_quantized_weights_one(
        record["soft_measurements"][test[0]], record["hard_measurements"][test[0]],
        *kernel_args, 0,
    )
    baseline_before = matching.decode_batch(detectors[:100])
    predictions: dict[int, np.ndarray] = {}
    for probability_bits in bits:
        values = np.empty(len(test), dtype=np.uint8)
        for position, row in enumerate(test):
            weights = fused_affine_quantized_weights_one(
                record["soft_measurements"][row], record["hard_measurements"][row],
                *kernel_args, probability_bits,
            )
            values[position] = matching.decode_preindexed(
                detectors[position], weights
            )[0]
        predictions[probability_bits] = values
        print(f"decoded {len(test)} records at bits={probability_bits}", flush=True)
    baseline_after = matching.decode_batch(detectors[:100])
    if np.any(baseline_before != baseline_after):
        raise RuntimeError("quantization campaign did not restore the base graph")
    reference = predictions[0]
    reference_errors = (reference != labels).astype(float)
    rows = []
    for probability_bits in bits:
        prediction = predictions[probability_bits]
        errors = (prediction != labels).astype(float)
        block_delta = (errors - reference_errors).reshape(-1, block_size).mean(axis=1)
        rows.append({
            "probability_bits": probability_bits,
            "logical_error": float(np.mean(errors)),
            "prediction_disagreements_vs_float": int(np.sum(prediction != reference)),
            "absolute_error_increase_vs_float": float(np.mean(errors - reference_errors)),
            "paired_block_95pct_t_interval_error_increase": (
                [0.0, 0.0] if probability_bits == 0 else interval(block_delta)
            ),
            "block_errors": errors.reshape(-1, block_size).mean(axis=1).tolist(),
        })
    commit = os.environ.get("PTWM_CODE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "fork_url": FORK_URL, "fork_commit": FORK_COMMIT,
            "pymatching_module": str(Path(pymatching.__file__).resolve()),
        },
        "data": {
            "file": path.name, "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            "circuit_group": circuit_group, "calibration_records": len(train),
            "benchmark_records": len(test), "block_size": block_size,
        },
        "design": {
            "quantizer": (
                "nearest of 2**b uniformly spaced reconstruction levels on [0,0.5], "
                "then clip to the existing soft-probability bounds; b=0 is float"
            ),
            "inspiration": "Hanisch et al., APS Open Science 1, 000019 (2026)",
            "primary_gate": (
                "8-bit upper paired 95% bound on logical-error increase <=0.001 in "
                "both Rigetti sessions; exploratory because both sessions were previously accessed"
            ),
            "timing_scope": "accuracy and information-volume screen; no hardware latency claim",
        },
        "graph": graph_diagnostics,
        "soft_reweighting": soft_diagnostics,
        "dynamic_edges": int(len(packed.residual_factor)),
        "base_restore_disagreements": int(np.sum(baseline_before != baseline_after)),
        "quantization": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", required=True)
    parser.add_argument("--records", type=int, default=40_000)
    parser.add_argument("--bits", type=int, nargs="+", default=[0, 8, 6, 4, 3, 2])
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        args.data, args.circuit_group, args.records, args.bits, args.block_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "data": payload["data"], "quantization": payload["quantization"],
        "base_restore_disagreements": payload["base_restore_disagreements"],
    }, indent=2))
