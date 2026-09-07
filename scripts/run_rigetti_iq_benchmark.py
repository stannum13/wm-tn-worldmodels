#!/usr/bin/env python3
"""Chronological real-hardware I/Q calibration screen on Rigetti Ankaa-2."""

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
    block_score_differences,
    chronological_preparation_split,
    fit_iq_heads,
    load_measurement_fidelity,
    probability_metrics,
    timed_head_prediction,
)


def _interval(values: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def run(path: Path) -> dict:
    features, hard, labels = load_measurement_fidelity(str(path))
    train, test = chronological_preparation_split(labels, train_fraction=0.6)
    heads = fit_iq_heads(features[train], labels[train])
    hard_probability = {
        bit: float(np.mean(labels[train][hard[train] == bit])) for bit in (0.0, 1.0)
    }
    predictions = {
        "calibrated_hardware_hard_bit": np.asarray([hard_probability[bit] for bit in hard[test]])
    }
    timing = {"calibrated_hardware_hard_bit": {"vectorized_ns_per_sample": 0.0}}
    for name, head in heads.items():
        predictions[name], timing[name] = timed_head_prediction(head, features[test])
    rows = []
    for name, probability in predictions.items():
        parameters = 2 if name == "calibrated_hardware_hard_bit" else int(heads[name].weights.size)
        rows.append({"model": name, "parameters": parameters, **probability_metrics(probability, labels[test]), **timing[name]})
    comparisons = []
    for first, second in (
        ("calibrated_hardware_hard_bit", "linear_logistic_iq"),
        ("linear_logistic_iq", "spline_kan_logistic_iq"),
    ):
        error_differences = block_error_differences(
            predictions[first], predictions[second], labels[test], block_size=200
        )
        brier_differences = block_score_differences(
            predictions[first], predictions[second], labels[test], block_size=200,
            score="brier_loss",
        )
        nll_differences = block_score_differences(
            predictions[first], predictions[second], labels[test], block_size=200,
            score="negative_log_likelihood",
        )
        comparisons.append({
            "first": first,
            "second": second,
            "mean_error_reduction": float(np.mean(error_differences)),
            "error_reduction_paired_95pct_t_interval": _interval(error_differences),
            "mean_brier_reduction": float(np.mean(brier_differences)),
            "brier_reduction_paired_95pct_t_interval": _interval(brier_differences),
            "mean_nll_reduction": float(np.mean(nll_differences)),
            "nll_reduction_paired_95pct_t_interval": _interval(nll_differences),
            "blocks": len(error_differences),
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
            "qpu": "Rigetti Ankaa-2",
            "train_shots": len(train),
            "chronological_test_shots": len(test),
            "independent_sessions": 1,
        },
        "design": {
            "endpoint": "prepared-state discrimination from raw complex I/Q",
            "split": "first 60% within each preparation for training; final 40% locked test",
            "uncertainty": "paired descriptive t interval across contiguous 200-shot blocks within preparation",
            "scope": "within-session calibration screen, not a QEC decoding or cross-session generalization claim",
        },
        "models": rows,
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/rigetti_fast_feedback/fast_feedback_raw_data.h5"))
    parser.add_argument("--output", type=Path, default=Path("results/rigetti_iq_benchmark.json"))
    args = parser.parse_args()
    payload = run(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"models": payload["models"], "comparisons": payload["comparisons"]}, indent=2))
