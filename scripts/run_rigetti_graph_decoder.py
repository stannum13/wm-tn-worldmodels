#!/usr/bin/env python3
"""Test cheap circuit-local state features on the Ankaa-2 stability record."""

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
    chronological_block_error_differences,
    chronological_block_score_differences,
    detector_worldline_parities,
    fit_logistic_head,
    load_qec_record,
    local_detector_pairs,
    local_pair_features,
    probability_metrics,
    timed_head_prediction,
)


def _interval(values: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def run(path: Path, circuit_group: str, block_size: int) -> dict:
    record = load_qec_record(str(path), circuit_group)
    detectors = record["detectors"].astype(float)
    labels = record["observables"][:, 0].astype(float)
    boundary = int(0.6 * len(labels))
    train, test = np.arange(boundary), np.arange(boundary, len(labels))
    pairs = local_detector_pairs(record["circuit"])
    parity = detector_worldline_parities(detectors, record["circuit"])
    pair_features = local_pair_features(detectors, pairs)
    feature_sets = {
        "linear_detector": detectors,
        "worldline_parity": np.c_[detectors, parity],
        "local_pair": np.c_[detectors, pair_features],
        "worldline_plus_local_pair": np.c_[detectors, parity, pair_features],
    }
    rows, predictions = [], {}
    for name, features in feature_sets.items():
        head = fit_logistic_head(features[train], labels[train], knots=1)
        prediction, timing = timed_head_prediction(head, features[test])
        predictions[name] = prediction
        rows.append({
            "model": name,
            "parameters": len(head.weights),
            **probability_metrics(prediction, labels[test]),
            **timing,
        })
    comparisons = []
    for second in ("worldline_parity", "local_pair", "worldline_plus_local_pair"):
        error = chronological_block_error_differences(predictions["linear_detector"], predictions[second], labels[test], block_size=block_size)
        brier = chronological_block_score_differences(predictions["linear_detector"], predictions[second], labels[test], block_size=block_size, score="brier_loss")
        comparisons.append({
            "first": "linear_detector",
            "second": second,
            "mean_error_reduction": float(np.mean(error)),
            "error_reduction_paired_95pct_t_interval": _interval(error),
            "mean_brier_reduction": float(np.mean(brier)),
            "brier_reduction_paired_95pct_t_interval": _interval(brier),
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
            "shots": len(labels),
            "train_shots": len(train),
            "row_order_test_shots": len(test),
            "independent_sessions": 1,
        },
        "design": {
            "local_pairs": len(pairs),
            "pair_rule": "detectors within one round and Manhattan spatial distance <=2.01; no target-informed selection",
            "released_mwpm_error_at_27_rounds": 0.38819,
            "primary_go_condition": "close >=20% of linear-to-released-MWPM error gap with positive paired block interval",
            "split": "first 60% HDF5 rows train; final 40% test; per-shot timestamps absent",
            "limitations": "single-session learned replay; released MWPM value is an external anchor, not rerun in this script",
        },
        "models": rows,
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", required=True)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results/rigetti_graph_decoder.json"))
    args = parser.parse_args()
    payload = run(args.data, args.circuit_group, args.block_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"models": payload["models"], "comparisons": payload["comparisons"]}, indent=2))
