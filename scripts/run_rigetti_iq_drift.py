#!/usr/bin/env python3
"""Test whether causal I/Q recalibration tracks useful row-order drift."""

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
from ptwm.rigetti import fit_logistic_head, load_qec_record  # noqa: E402


def _balanced_head(values: np.ndarray, targets: np.ndarray, cap: int = 100_000):
    locations = [np.flatnonzero(targets == label) for label in (0.0, 1.0)]
    per_class = min(*(len(item) for item in locations), cap // 2)
    selected = np.concatenate([
        item[np.linspace(0, len(item) - 1, per_class, dtype=int)] for item in locations
    ])
    features = np.c_[values.real, values.imag]
    return fit_logistic_head(features[selected], targets[selected], knots=1)


def _predict_heads(record: dict[str, object], rows: np.ndarray, heads: dict[int, object]) -> np.ndarray:
    output = np.empty((len(rows), record["hard_measurements"].shape[1]), dtype=float)
    for qubit, head in heads.items():
        columns = np.flatnonzero(record["measurement_qubits"] == qubit)
        values = record["soft_measurements"][np.ix_(rows, columns)].reshape(-1)
        output[:, columns] = head.predict(np.c_[values.real, values.imag]).reshape(len(rows), -1)
    return output


def _fit_heads(record: dict[str, object], rows: np.ndarray) -> dict[int, object]:
    heads = {}
    for qubit in np.unique(record["measurement_qubits"]):
        columns = np.flatnonzero(record["measurement_qubits"] == qubit)
        values = record["soft_measurements"][np.ix_(rows, columns)].reshape(-1)
        targets = record["hard_measurements"][np.ix_(rows, columns)].reshape(-1).astype(float)
        heads[int(qubit)] = _balanced_head(values, targets)
    return heads


def _scores(probability: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return (
        float(np.mean((probability - targets) ** 2)),
        float(-np.mean(targets * np.log(probability) + (1 - targets) * np.log(1 - probability))),
    )


def _interval(values: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def run(path: Path, circuit_group: str, window: int, update_rows: int) -> dict:
    record = load_qec_record(str(path), circuit_group)
    static_heads = _fit_heads(record, np.arange(window))
    blocks, static_brier, rolling_brier, static_nll, rolling_nll = [], [], [], [], []
    state_vectors = []
    for start in range(window, len(record["hard_measurements"]) - update_rows + 1, update_rows):
        stop = start + update_rows
        evaluation = np.arange(start, stop)
        rolling_heads = _fit_heads(record, np.arange(start - window, start))
        static_probability = _predict_heads(record, evaluation, static_heads)
        rolling_probability = _predict_heads(record, evaluation, rolling_heads)
        targets = record["hard_measurements"][evaluation].astype(float)
        sb, sn = _scores(static_probability, targets)
        rb, rn = _scores(rolling_probability, targets)
        static_brier.append(sb)
        rolling_brier.append(rb)
        static_nll.append(sn)
        rolling_nll.append(rn)
        state_vectors.append(np.concatenate([rolling_heads[q].weights for q in sorted(rolling_heads)]))
        blocks.append({
            "start_row": start, "stop_row": stop,
            "static_brier": sb, "rolling_brier": rb,
            "static_nll": sn, "rolling_nll": rn,
        })
        print(f"calibrated through row {stop}", flush=True)
    brier_difference = np.asarray(static_brier) - np.asarray(rolling_brier)
    nll_difference = np.asarray(static_nll) - np.asarray(rolling_nll)
    states = np.asarray(state_vectors)
    centered = states - states.mean(axis=0)
    lag_correlation = np.sum(centered[:-1] * centered[1:], axis=0) / np.sqrt(
        np.sum(centered[:-1] ** 2, axis=0) * np.sum(centered[1:] ** 2, axis=0)
    )
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
        },
        "design": {
            "initial_rows": window, "rolling_window_rows": window,
            "update_rows": update_rows,
            "target": "recorded hardware hard decision; logical labels excluded",
            "go_condition": ">=1% NLL reduction with positive paired block interval",
            "routing": "run robust EKF only after GO; particles only after residual multimodality",
            "limitation": "HDF5 row-order drift diagnostic; per-shot timestamps unavailable",
        },
        "summary": {
            "static_brier": float(np.mean(static_brier)),
            "rolling_brier": float(np.mean(rolling_brier)),
            "relative_brier_reduction": float(1 - np.mean(rolling_brier) / np.mean(static_brier)),
            "brier_reduction_paired_95pct_t_interval": _interval(brier_difference),
            "static_nll": float(np.mean(static_nll)),
            "rolling_nll": float(np.mean(rolling_nll)),
            "relative_nll_reduction": float(1 - np.mean(rolling_nll) / np.mean(static_nll)),
            "nll_reduction_paired_95pct_t_interval": _interval(nll_difference),
            "blocks": len(blocks),
            "median_parameter_lag1_correlation": float(np.nanmedian(lag_correlation)),
        },
        "blocks": blocks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", default="circuit_22")
    parser.add_argument("--window", type=int, default=20_000)
    parser.add_argument("--update-rows", type=int, default=5_000)
    parser.add_argument("--output", type=Path, default=Path("results/rigetti_iq_drift.json"))
    args = parser.parse_args()
    payload = run(args.data, args.circuit_group, args.window, args.update_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2))
