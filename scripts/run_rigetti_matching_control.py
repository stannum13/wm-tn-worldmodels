#!/usr/bin/env python3
"""Run an auditable PyMatching control on the real Ankaa-2 syndrome record."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.rigetti import (  # noqa: E402
    block_error_differences,
    load_qec_record,
    matching_predictions,
    probability_metrics,
)


def _interval(values: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def _calibrate_hard_prediction(prediction: np.ndarray, labels: np.ndarray) -> dict[float, float]:
    return {
        bit: float((np.sum(labels[prediction == bit]) + 1.0) / (np.sum(prediction == bit) + 2.0))
        for bit in (0.0, 1.0)
    }


def run(path: Path, circuit_group: str, probabilities: list[float], block_size: int) -> dict:
    record = load_qec_record(str(path), circuit_group)
    detectors = record["detectors"]
    labels = record["observables"][:, 0].astype(float)
    boundary = int(0.6 * len(labels))
    train, test = np.arange(boundary), np.arange(boundary, len(labels))
    development = []
    for probability in probabilities:
        prediction, info = matching_predictions(
            record["circuit"], detectors[train], probability=probability
        )
        development.append({
            "circuit_error_probability": probability,
            "train_logical_error": float(np.mean(prediction != labels[train])),
            "detector_error_model_terms": info["detector_error_model_terms"],
        })
    selected = min(development, key=lambda row: row["train_logical_error"])["circuit_error_probability"]
    train_prediction, _ = matching_predictions(record["circuit"], detectors[train], probability=selected)
    reliability = _calibrate_hard_prediction(train_prediction, labels[train])
    test_prediction, timing = matching_predictions(record["circuit"], detectors[test], probability=selected)
    probability = np.asarray([reliability[bit] for bit in test_prediction])
    errors = []
    for start in range(0, len(test) - block_size + 1, block_size):
        block = slice(start, start + block_size)
        errors.append(np.mean(test_prediction[block] != labels[test][block]))
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
            "shots": len(labels),
            "train_shots": len(train),
            "chronological_test_shots": len(test),
            "independent_sessions": 1,
        },
        "design": {
            "noise_model": "uniform measurement flips, reset flips, one-qubit depolarization, and CZ depolarization inserted into embedded circuit",
            "selection": "circuit error probability selected only by chronological training error",
            "released_mwpm_error_at_27_rounds": 0.38819,
            "limitation": "transparent topology control, not the unpublished calibrated detector error model used for the released result",
        },
        "development": development,
        "selected_circuit_error_probability": selected,
        "test": {
            **probability_metrics(probability, labels[test]),
            "hard_logical_error": float(np.mean(test_prediction != labels[test])),
            "hard_error_block_95pct_t_interval": _interval(np.asarray(errors)),
            "reliability_probabilities": reliability,
            **timing,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", required=True)
    parser.add_argument("--probabilities", default="0.002,0.005,0.01,0.02,0.05")
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results/rigetti_matching_control.json"))
    args = parser.parse_args()
    payload = run(
        args.data, args.circuit_group,
        [float(value) for value in args.probabilities.split(",")], args.block_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"development": payload["development"], "test": payload["test"]}, indent=2))
