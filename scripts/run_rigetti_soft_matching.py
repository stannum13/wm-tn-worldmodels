#!/usr/bin/env python3
"""Cross-fitted I/Q reweighting of a pairwise-correlation matching graph."""

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
    chronological_block_error_differences,
    fit_spitz_pairwise_matching,
    load_qec_record,
    measurement_error_signatures,
    prepare_soft_reweighting,
    typed_circuit_noise_model,
)


def _interval(values: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def run(
    path: Path, circuit_group: str, train_fraction: float, block_size: int,
    max_test: int | None, knots: int,
) -> dict:
    import pymatching

    record = load_qec_record(str(path), circuit_group)
    labels = record["observables"][:, 0].astype(float)
    boundary = int(train_fraction * len(labels))
    train = np.arange(boundary)
    test_stop = len(labels) if max_test is None else min(len(labels), boundary + max_test)
    test = np.arange(boundary, test_stop)

    measurement_probability, parameters = calibrate_measurement_probabilities(
        record["soft_measurements"], record["hard_measurements"],
        record["measurement_qubits"], train, knots=knots, balanced=True,
    )
    measurement_error = np.where(
        record["hard_measurements"], 1.0 - measurement_probability, measurement_probability
    )
    template_dem = typed_circuit_noise_model(
        record["circuit"], measurement_probability=0.03,
        one_qubit_probability=0.003, two_qubit_probability=0.03,
    )
    template = pymatching.Matching.from_detector_error_model(template_dem)
    pairwise, graph_diagnostics = fit_spitz_pairwise_matching(
        template, record["detectors"][train], floor_probability=1e-4
    )
    signatures = measurement_error_signatures(record["circuit"])
    plan, soft_diagnostics = prepare_soft_reweighting(
        pairwise, signatures, np.mean(measurement_error[train], axis=0)
    )

    template_prediction = np.asarray(
        template.decode_batch(record["detectors"][test].astype(np.uint8))
    )[:, 0]
    pairwise_prediction = np.asarray(
        pairwise.decode_batch(record["detectors"][test].astype(np.uint8))
    )[:, 0]
    soft_prediction = np.empty(len(test))
    call_ns = np.empty(len(test))
    for position, row in enumerate(test):
        started = perf_counter_ns()
        matching = build_soft_reweighted_matching(plan, measurement_error[row])
        soft_prediction[position] = matching.decode(
            record["detectors"][row].astype(np.uint8)
        )[0]
        call_ns[position] = perf_counter_ns() - started
        if (position + 1) % 5000 == 0:
            print(f"decoded {position + 1}/{len(test)} soft shots", flush=True)

    soft_name = "pairwise_linear_iq" if knots == 1 else f"pairwise_spline_iq_k{knots}"
    predictions = {
        "circuit_template_hard": template_prediction,
        "pairwise_hard": pairwise_prediction,
        soft_name: soft_prediction,
    }
    rows = [{
        "model": name,
        "logical_error": float(np.mean(prediction != labels[test])),
    } for name, prediction in predictions.items()]
    comparisons = []
    for first in ("circuit_template_hard", "pairwise_hard"):
        differences = chronological_block_error_differences(
            predictions[first], soft_prediction, labels[test], block_size=block_size
        )
        first_error = float(np.mean(predictions[first] != labels[test]))
        soft_error = float(np.mean(soft_prediction != labels[test]))
        comparisons.append({
            "first": first,
            "second": soft_name,
            "mean_absolute_error_reduction": first_error - soft_error,
            "relative_error_reduction": 1.0 - soft_error / first_error,
            "paired_row_block_95pct_t_interval": _interval(differences),
            "blocks": len(differences),
        })
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
            "source": "https://zenodo.org/records/15364358",
            "file": path.name,
            "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            "circuit_group": circuit_group,
            "train_rows": len(train),
            "test_rows": len(test),
        },
        "design": {
            "split": "I/Q calibration and graph fitting use first HDF5 rows; final rows are locked",
            "calibration": f"balanced logistic P(hardware hard bit | I,Q) per qubit with {knots} spline-design knots",
            "soft_rule": "replace average measurement contribution on matched edges with per-shot posterior error",
            "logical_labels_in_iq_or_graph_calibration": False,
            "primary_go_condition": ">=1% relative logical-error reduction versus both hard controls with positive paired intervals",
            "limitation": "hard decisions are pseudo-labels because authors' prepared-state calibration is not released; unmatched measurement signatures remain hard-weighted",
        },
        "calibration_parameters": parameters,
        "graph": graph_diagnostics,
        "soft_reweighting": soft_diagnostics,
        "models": rows,
        "row_block_errors": {
            name: (prediction != labels[test]).reshape(-1, block_size).mean(axis=1).tolist()
            for name, prediction in predictions.items()
        },
        "comparisons": comparisons,
        "soft_build_and_decode_latency": {
            "python_batch_one_p50_ns": float(np.quantile(call_ns, 0.50)),
            "python_batch_one_p99_ns": float(np.quantile(call_ns, 0.99)),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", default="circuit_22")
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--knots", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/rigetti_soft_matching.json"))
    args = parser.parse_args()
    payload = run(
        args.data, args.circuit_group, args.train_fraction, args.block_size,
        args.max_test, args.knots,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "models": payload["models"], "comparisons": payload["comparisons"],
        "soft_reweighting": payload["soft_reweighting"],
        "soft_build_and_decode_latency": payload["soft_build_and_decode_latency"],
    }, indent=2))
