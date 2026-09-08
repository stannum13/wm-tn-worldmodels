#!/usr/bin/env python3
"""Test whether causal affine I/Q recalibration mediates logical-error gains."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.rigetti import (  # noqa: E402
    build_soft_reweighted_matching,
    calibrate_measurement_probabilities,
    fit_spitz_pairwise_matching,
    load_qec_record,
    measurement_error_signatures,
    prepare_soft_reweighting,
    typed_circuit_noise_model,
)


def _errors(probability: np.ndarray, hard: np.ndarray) -> np.ndarray:
    return np.where(hard, 1.0 - probability, probability)


def _interval(values: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def run(path: Path, circuit_group: str, train_rows: int, window: int, block_size: int) -> dict:
    import pymatching

    record = load_qec_record(str(path), circuit_group)
    labels = record["observables"][:, 0].astype(float)
    hard = record["hard_measurements"]
    initial = np.arange(train_rows)
    static_probability, parameters = calibrate_measurement_probabilities(
        record["soft_measurements"], hard, record["measurement_qubits"],
        initial, knots=1, balanced=True,
    )
    static_error = _errors(static_probability, hard)
    template_dem = typed_circuit_noise_model(
        record["circuit"], measurement_probability=0.03,
        one_qubit_probability=0.003, two_qubit_probability=0.03,
    )
    template = pymatching.Matching.from_detector_error_model(template_dem)
    pairwise, graph_diagnostics = fit_spitz_pairwise_matching(
        template, record["detectors"][initial], floor_probability=1e-4
    )
    signatures = measurement_error_signatures(record["circuit"])
    static_plan, static_diagnostics = prepare_soft_reweighting(
        pairwise, signatures, np.mean(static_error[initial], axis=0)
    )

    static_prediction, rolling_prediction = [], []
    update_ns, block_rows = [], []
    for start in range(train_rows, len(labels) - block_size + 1, block_size):
        stop = start + block_size
        calibration = np.arange(start - window, start)
        began = perf_counter_ns()
        rolling_probability, _ = calibrate_measurement_probabilities(
            record["soft_measurements"], hard, record["measurement_qubits"],
            calibration, knots=1, balanced=True,
        )
        rolling_error = _errors(rolling_probability, hard)
        rolling_plan, rolling_diagnostics = prepare_soft_reweighting(
            pairwise, signatures, np.mean(rolling_error[calibration], axis=0)
        )
        update_ns.append(perf_counter_ns() - began)
        static_block, rolling_block = [], []
        for row in range(start, stop):
            static_matching = build_soft_reweighted_matching(static_plan, static_error[row])
            rolling_matching = build_soft_reweighted_matching(rolling_plan, rolling_error[row])
            detector = record["detectors"][row].astype(np.uint8)
            static_block.append(static_matching.decode(detector)[0])
            rolling_block.append(rolling_matching.decode(detector)[0])
        static_prediction.extend(static_block)
        rolling_prediction.extend(rolling_block)
        target = labels[start:stop]
        block_rows.append({
            "start_row": start, "stop_row": stop,
            "static_error": float(np.mean(np.asarray(static_block) != target)),
            "rolling_error": float(np.mean(np.asarray(rolling_block) != target)),
            "rolling_residual_clipped": rolling_diagnostics["residual_clipped"],
        })
        print(f"decoded through row {stop}", flush=True)
    static_prediction = np.asarray(static_prediction)
    rolling_prediction = np.asarray(rolling_prediction)
    targets = labels[train_rows : train_rows + len(static_prediction)]
    static_rate = float(np.mean(static_prediction != targets))
    rolling_rate = float(np.mean(rolling_prediction != targets))
    differences = np.asarray([
        row["static_error"] - row["rolling_error"] for row in block_rows
    ])
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "data": {
            "source": "https://zenodo.org/records/15364358", "file": path.name,
            "md5": hashlib.md5(path.read_bytes()).hexdigest(), "circuit_group": circuit_group,
            "train_rows": train_rows, "test_rows": len(targets),
        },
        "design": {
            "fixed_graph": "Spitz pairwise graph fitted on the initial row range",
            "static_calibrator": "balanced affine I/Q head fitted on the initial row range",
            "rolling_calibrator": "balanced affine I/Q head fitted only on the preceding row window",
            "rolling_window_rows": window, "update_rows": block_size,
            "logical_labels_in_calibration": False,
            "go_condition": ">=1% relative logical-error reduction with positive paired block interval",
            "status": "exploratory mediation test on an accessed acquisition",
        },
        "calibration_parameters": parameters,
        "graph": graph_diagnostics,
        "static_soft_reweighting": static_diagnostics,
        "result": {
            "static_logical_error": static_rate,
            "rolling_logical_error": rolling_rate,
            "absolute_error_reduction": static_rate - rolling_rate,
            "relative_error_reduction": 1.0 - rolling_rate / static_rate,
            "paired_row_block_95pct_t_interval": _interval(differences),
            "blocks": len(differences),
        },
        "slow_update": {
            "p50_ms": float(np.quantile(update_ns, 0.50) / 1e6),
            "p99_ms": float(np.quantile(update_ns, 0.99) / 1e6),
        },
        "blocks": block_rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", default="circuit_22")
    parser.add_argument("--train-rows", type=int, default=60_000)
    parser.add_argument("--window", type=int, default=20_000)
    parser.add_argument("--block-size", type=int, default=5_000)
    parser.add_argument("--output", type=Path, default=Path("results/rigetti_adaptive_soft_matching.json"))
    args = parser.parse_args()
    payload = run(args.data, args.circuit_group, args.train_rows, args.window, args.block_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": payload["result"], "slow_update": payload["slow_update"]}, indent=2))
