#!/usr/bin/env python3
"""Prototype an exact temporal-frontier decoder against PyMatching reconstruction."""

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
from time import perf_counter_ns

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.rigetti import (  # noqa: E402
    build_soft_reweighted_matching,
    calibrate_measurement_probabilities,
    compile_frontier_decoder,
    fit_spitz_pairwise_matching,
    load_qec_record,
    measurement_error_signatures,
    pack_full_edge_weights,
    prepare_soft_reweighting,
    typed_circuit_noise_model,
)
from ptwm.rigetti_jit import (  # noqa: E402
    decode_frontier_batch,
    decode_frontier_one,
    pack_frontier_schedule,
)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50_ns": float(np.quantile(values, 0.50)),
        "p95_ns": float(np.quantile(values, 0.95)),
        "p99_ns": float(np.quantile(values, 0.99)),
    }


def run(path: Path, circuit_group: str, train_fraction: float, max_test: int) -> dict:
    import pymatching

    record = load_qec_record(str(path), circuit_group)
    labels = record["observables"][:, 0].astype(np.uint8)
    boundary = int(train_fraction * len(labels))
    train = np.arange(boundary)
    test = np.arange(boundary, min(len(labels), boundary + max_test))
    probability, parameters = calibrate_measurement_probabilities(
        record["soft_measurements"], record["hard_measurements"],
        record["measurement_qubits"], train, knots=1, balanced=True,
    )
    measurement_error = np.where(
        record["hard_measurements"], 1.0 - probability, probability
    )
    template = pymatching.Matching.from_detector_error_model(typed_circuit_noise_model(
        record["circuit"], measurement_probability=0.03,
        one_qubit_probability=0.003, two_qubit_probability=0.03,
    ))
    pairwise, graph_diagnostics = fit_spitz_pairwise_matching(
        template, record["detectors"][train], floor_probability=1e-4
    )
    plan, soft_diagnostics = prepare_soft_reweighting(
        pairwise, measurement_error_signatures(record["circuit"]),
        np.mean(measurement_error[train], axis=0),
    )
    frontier = compile_frontier_decoder(
        pairwise, record["circuit"].get_detector_coordinates()
    )
    full_weight_plan = pack_full_edge_weights(plan)
    weights = full_weight_plan.build(measurement_error[test])[:, :, 2]
    schedule = pack_frontier_schedule(frontier)

    reference = np.empty(len(test), dtype=np.uint8)
    reference_ns = np.empty(len(test))
    candidate = np.empty(len(test), dtype=np.uint8)
    candidate_ns = np.empty(len(test))
    margins = np.empty(len(test))
    for position, row in enumerate(test):
        started = perf_counter_ns()
        reference[position] = build_soft_reweighted_matching(
            plan, measurement_error[row]
        ).decode(record["detectors"][row].astype(np.uint8))[0]
        reference_ns[position] = perf_counter_ns() - started
        started = perf_counter_ns()
        candidate[position], margins[position] = frontier.decode(
            record["detectors"][row], weights[position]
        )
        candidate_ns[position] = perf_counter_ns() - started

    # Compile outside all recorded timing, then measure both batch-one latency and
    # amortized throughput for exactly the same DP schedule and weights.
    decode_frontier_one(record["detectors"][test[0]], weights[0], *schedule)
    compiled = np.empty(len(test), dtype=np.uint8)
    compiled_margins = np.empty(len(test))
    compiled_ns = np.empty(len(test))
    for position, row in enumerate(test):
        started = perf_counter_ns()
        compiled[position], compiled_margins[position] = decode_frontier_one(
            record["detectors"][row], weights[position], *schedule
        )
        compiled_ns[position] = perf_counter_ns() - started
    decode_frontier_batch(record["detectors"][test[:1]], weights[:1], *schedule)
    started = perf_counter_ns()
    compiled_batch, compiled_batch_margins = decode_frontier_batch(
        record["detectors"][test], weights, *schedule
    )
    compiled_batch_ns = perf_counter_ns() - started

    commit = os.environ.get("PTWM_CODE_COMMIT")
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
    disagreements = candidate != reference
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "data": {
            "source": "https://zenodo.org/records/15364358",
            "file": path.name,
            "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            "circuit_group": circuit_group,
            "train_rows": len(train),
            "test_rows": len(test),
        },
        "design": {
            "objective": "exact additive binary T-join with one logical parity bit",
            "order": "detector time, remaining Stim coordinates, node index",
            "reference": "fresh per-record PyMatching graph reconstruction",
            "prototype_go": "zero disagreements and p50 below reconstruction p50",
        },
        "calibration_parameters": parameters,
        "graph": graph_diagnostics,
        "soft_reweighting": soft_diagnostics,
        "frontier": {
            "maximum_active_separator": frontier.maximum_active_separator,
            "maximum_logical_parity_states": 2 ** (
                frontier.maximum_active_separator + 1
            ),
        },
        "prediction_disagreements": int(np.sum(disagreements)),
        "compiled_prediction_disagreements": int(np.sum(compiled != reference)),
        "compiled_batch_prediction_disagreements": int(np.sum(
            compiled_batch != reference
        )),
        "maximum_compiled_margin_difference": float(np.max(np.abs(
            compiled_margins - margins
        ))),
        "maximum_compiled_batch_margin_difference": float(np.max(np.abs(
            compiled_batch_margins - margins
        ))),
        "minimum_margin": float(np.min(margins)),
        "disagreement_margins": margins[disagreements].tolist(),
        "reference_logical_error": float(np.mean(reference != labels[test])),
        "candidate_logical_error": float(np.mean(candidate != labels[test])),
        "latency": {
            "reference_rebuild_and_decode": _quantiles(reference_ns),
            "python_frontier_decode": _quantiles(candidate_ns),
            "compiled_frontier_batch_one": _quantiles(compiled_ns),
            "compiled_frontier_batch_total_ns": float(compiled_batch_ns),
            "compiled_frontier_batch_amortized_ns_per_record": float(
                compiled_batch_ns / len(test)
            ),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--max-test", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.data, args.circuit_group, args.train_fraction, args.max_test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
