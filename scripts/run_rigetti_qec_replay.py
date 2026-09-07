#!/usr/bin/env python3
"""Replay Ankaa-2 stability-code records with hard and calibrated-soft decoders."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.rigetti import (  # noqa: E402
    chronological_block_error_differences,
    chronological_block_score_differences,
    calibrate_measurement_probabilities,
    detector_measurement_indices,
    fit_logistic_head,
    load_qec_record,
    probability_metrics,
    soft_detector_probabilities,
    timed_head_prediction,
)


def _interval(values: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def run(path: Path, circuit_group: str, train_fraction: float, block_size: int) -> dict:
    record = load_qec_record(str(path), circuit_group)
    hard = record["hard_measurements"]
    detectors = record["detectors"]
    labels = record["observables"][:, 0].astype(float)
    boundary = int(len(labels) * train_fraction)
    train, test = np.arange(boundary), np.arange(boundary, len(labels))
    predictions: dict[str, np.ndarray] = {}
    rows = []

    hard_decoder = fit_logistic_head(detectors[train].astype(float), labels[train], knots=1)
    prediction, timing = timed_head_prediction(hard_decoder, detectors[test].astype(float))
    predictions["hard_syndrome_linear_decoder"] = prediction
    rows.append({
        "model": "hard_syndrome_linear_decoder",
        "parameters": len(hard_decoder.weights),
        "offline_calibration_and_replay_seconds": 0.0,
        **probability_metrics(prediction, labels[test]),
        **timing,
    })

    detector_indices = detector_measurement_indices(record["circuit"])
    for calibration_name, knots in (("linear_iq", 1), ("spline_kan_iq", 6)):
        started = perf_counter()
        measurement_probability, calibration_parameters = calibrate_measurement_probabilities(
            record["soft_measurements"], hard, record["measurement_qubits"], train,
            knots=knots,
        )
        soft_detectors = soft_detector_probabilities(
            measurement_probability, detector_indices, hard, detectors
        )
        decoder = fit_logistic_head(soft_detectors[train], labels[train], knots=1)
        elapsed = perf_counter() - started
        prediction, timing = timed_head_prediction(decoder, soft_detectors[test])
        name = f"{calibration_name}_soft_syndrome_linear_decoder"
        predictions[name] = prediction
        rows.append({
            "model": name,
            "parameters": calibration_parameters + len(decoder.weights),
            "offline_calibration_and_replay_seconds": elapsed,
            **probability_metrics(prediction, labels[test]),
            **timing,
        })
        if knots == 6:
            spline_decoder = fit_logistic_head(soft_detectors[train], labels[train], knots=3)
            spline_prediction, spline_timing = timed_head_prediction(
                spline_decoder, soft_detectors[test]
            )
            spline_name = "spline_kan_iq_soft_syndrome_spline_decoder"
            predictions[spline_name] = spline_prediction
            rows.append({
                "model": spline_name,
                "parameters": calibration_parameters + len(spline_decoder.weights),
                "offline_calibration_and_replay_seconds": elapsed,
                **probability_metrics(spline_prediction, labels[test]),
                **spline_timing,
            })

    comparisons = []
    pairs = (
        ("hard_syndrome_linear_decoder", "linear_iq_soft_syndrome_linear_decoder"),
        ("linear_iq_soft_syndrome_linear_decoder", "spline_kan_iq_soft_syndrome_linear_decoder"),
        ("spline_kan_iq_soft_syndrome_linear_decoder", "spline_kan_iq_soft_syndrome_spline_decoder"),
    )
    for first, second in pairs:
        error = chronological_block_error_differences(predictions[first], predictions[second], labels[test], block_size=block_size)
        brier = chronological_block_score_differences(predictions[first], predictions[second], labels[test], block_size=block_size, score="brier_loss")
        nll = chronological_block_score_differences(predictions[first], predictions[second], labels[test], block_size=block_size, score="negative_log_likelihood")
        comparisons.append({
            "first": first,
            "second": second,
            "mean_error_reduction": float(np.mean(error)),
            "error_reduction_paired_95pct_t_interval": _interval(error),
            "mean_brier_reduction": float(np.mean(brier)),
            "brier_reduction_paired_95pct_t_interval": _interval(brier),
            "mean_nll_reduction": float(np.mean(nll)),
            "nll_reduction_paired_95pct_t_interval": _interval(nll),
            "blocks": len(error),
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
            "qpu": "Rigetti Ankaa-2",
            "shots": len(labels),
            "measurements_per_shot": hard.shape[1],
            "detectors_per_shot": detectors.shape[1],
            "train_shots": len(train),
            "row_order_test_shots": len(test),
            "independent_sessions": 1,
        },
        "design": {
            "target": "logical observable reconstructed by the embedded Stim circuit",
            "split": f"first {train_fraction:.0%} HDF5 rows train; remaining rows test; per-shot timestamps absent",
            "calibration_target": "hardware hard bits; soft I/Q is never given the logical target directly",
            "uncertainty": f"descriptive paired t intervals across contiguous {block_size}-shot test blocks",
            "go_condition": "at least 5% relative Brier reduction with positive block interval and no >0.2 percentage-point error degradation",
            "limitations": "single session; learned replay, not prospective FPGA feedback",
        },
        "models": rows,
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results/rigetti_qec_replay.json"))
    args = parser.parse_args()
    payload = run(args.data, args.circuit_group, args.train_fraction, args.block_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"models": payload["models"], "comparisons": payload["comparisons"]}, indent=2))
