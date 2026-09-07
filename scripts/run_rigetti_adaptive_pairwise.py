#!/usr/bin/env python3
"""Causal rolling pairwise graph calibration with a fixed PyMatching hot path."""

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
    fit_spitz_pairwise_matching,
    load_qec_record,
    typed_circuit_noise_model,
)


def _interval(values: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def _decode(matching: object, detectors: np.ndarray) -> tuple[np.ndarray, float]:
    started = perf_counter_ns()
    prediction = np.asarray(matching.decode_batch(detectors.astype(np.uint8)))[:, 0]
    return prediction, (perf_counter_ns() - started) / len(detectors)


def run(path: Path, circuit_group: str, window: int, block_size: int, floor: float) -> dict:
    import pymatching

    record = load_qec_record(str(path), circuit_group)
    detectors = record["detectors"]
    labels = record["observables"][:, 0].astype(float)
    if window < block_size or len(labels) <= window:
        raise ValueError("window must contain at least one block and leave evaluation rows")
    template_dem = typed_circuit_noise_model(
        record["circuit"], measurement_probability=0.03,
        one_qubit_probability=0.003, two_qubit_probability=0.03,
    )
    template = pymatching.Matching.from_detector_error_model(template_dem)
    static, static_diagnostics = fit_spitz_pairwise_matching(
        template, detectors[:window], floor_probability=floor
    )

    predictions = {
        "template": np.empty(len(labels) - window),
        "static_pairwise": np.empty(len(labels) - window),
        "rolling_pairwise": np.empty(len(labels) - window),
    }
    update_ns, rolling_decode_ns, block_rows = [], [], []
    cursor = 0
    for start in range(window, len(labels) - block_size + 1, block_size):
        stop = start + block_size
        block = detectors[start:stop]
        predictions["template"][cursor : cursor + block_size], _ = _decode(template, block)
        predictions["static_pairwise"][cursor : cursor + block_size], _ = _decode(static, block)
        began = perf_counter_ns()
        rolling, diagnostics = fit_spitz_pairwise_matching(
            template, detectors[start - window : start], floor_probability=floor
        )
        update_ns.append(perf_counter_ns() - began)
        predictions["rolling_pairwise"][cursor : cursor + block_size], latency = _decode(rolling, block)
        rolling_decode_ns.append(latency)
        block_rows.append({
            "start_row": start,
            "stop_row": stop,
            "rolling_invalid_estimates": diagnostics["invalid_estimates"],
            "rolling_clipped_low": diagnostics["clipped_low"],
        })
        cursor += block_size
    evaluation_labels = labels[window : window + cursor]
    predictions = {name: value[:cursor] for name, value in predictions.items()}
    errors = {
        name: (prediction != evaluation_labels).reshape(-1, block_size).mean(axis=1)
        for name, prediction in predictions.items()
    }

    comparisons = []
    for first in ("template", "static_pairwise"):
        differences = errors[first] - errors["rolling_pairwise"]
        comparisons.append({
            "first": first,
            "second": "rolling_pairwise",
            "mean_absolute_error_reduction": float(np.mean(differences)),
            "relative_error_reduction": float(
                1.0 - np.mean(errors["rolling_pairwise"]) / np.mean(errors[first])
            ),
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
            "shots": len(labels),
            "evaluation_rows": cursor,
        },
        "design": {
            "calibration_window_rows": window,
            "update_rows": block_size,
            "causal_rule": "block b uses only the preceding rolling HDF5 row window",
            "labels_in_calibration": False,
            "per_shot_timestamps_available": False,
            "status": "exploratory on an accessed acquisition",
            "go_condition": ">=1% relative error reduction versus both static controls with positive paired intervals",
            "limitation": "row-order responsivity, not verified wall-clock drift; template omits idle-location faults",
        },
        "models": [
            {"model": name, "logical_error": float(np.mean(value))}
            for name, value in errors.items()
        ],
        "comparisons": comparisons,
        "slow_update": {
            "updates": len(update_ns),
            "p50_ms": float(np.quantile(update_ns, 0.50) / 1e6),
            "p99_ms": float(np.quantile(update_ns, 0.99) / 1e6),
            "amortized_ns_per_shot": float(np.sum(update_ns) / cursor),
            "static_graph": static_diagnostics,
        },
        "hot_decode": {
            "rolling_vectorized_p50_ns_per_shot": float(np.quantile(rolling_decode_ns, 0.50)),
            "rolling_vectorized_p99_ns_per_shot": float(np.quantile(rolling_decode_ns, 0.99)),
        },
        "blocks": block_rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", default="circuit_22")
    parser.add_argument("--window", type=int, default=20_000)
    parser.add_argument("--block-size", type=int, default=1_000)
    parser.add_argument("--floor-probability", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("results/rigetti_adaptive_pairwise.json"))
    args = parser.parse_args()
    payload = run(args.data, args.circuit_group, args.window, args.block_size, args.floor_probability)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "models": payload["models"], "comparisons": payload["comparisons"],
        "slow_update": payload["slow_update"], "hot_decode": payload["hot_decode"],
    }, indent=2))
