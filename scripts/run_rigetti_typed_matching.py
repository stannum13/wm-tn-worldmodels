#!/usr/bin/env python3
"""Calibrate three circuit-noise rates for the Ankaa-2 matching graph."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
    typed_matching_predictions,
)


def _interval(values: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def run(path: Path, circuit_group: str, rates: list[float], block_size: int) -> dict:
    record = load_qec_record(str(path), circuit_group)
    detectors = record["detectors"]
    labels = record["observables"][:, 0].astype(float)
    development, validation, test = np.arange(20_000), np.arange(20_000, 60_000), np.arange(60_000, len(labels))
    coarse = []
    for measurement, one_qubit, two_qubit in itertools.product(rates, repeat=3):
        prediction, info = typed_matching_predictions(
            record["circuit"], detectors[development],
            measurement_probability=measurement,
            one_qubit_probability=one_qubit,
            two_qubit_probability=two_qubit,
        )
        coarse.append({
            "measurement_probability": measurement,
            "one_qubit_probability": one_qubit,
            "two_qubit_probability": two_qubit,
            "development_error": float(np.mean(prediction != labels[development])),
            "development_ns_per_shot": info["vectorized_ns_per_shot"],
        })
    finalists = sorted(coarse, key=lambda row: row["development_error"])[:5]
    for candidate in finalists:
        prediction, _ = typed_matching_predictions(
            record["circuit"], detectors[validation],
            measurement_probability=candidate["measurement_probability"],
            one_qubit_probability=candidate["one_qubit_probability"],
            two_qubit_probability=candidate["two_qubit_probability"],
        )
        candidate["validation_error"] = float(np.mean(prediction != labels[validation]))
    selected = min(finalists, key=lambda row: row["validation_error"])
    typed, typed_timing = typed_matching_predictions(
        record["circuit"], detectors[test],
        measurement_probability=selected["measurement_probability"],
        one_qubit_probability=selected["one_qubit_probability"],
        two_qubit_probability=selected["two_qubit_probability"],
    )
    uniform, uniform_timing = matching_predictions(
        record["circuit"], detectors[test], probability=0.002
    )
    differences = block_error_differences(uniform, typed, labels[test], block_size=block_size)
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
            "development_shots": len(development),
            "validation_shots": len(validation),
            "locked_test_shots": len(test),
            "independent_sessions": 1,
        },
        "design": {
            "coarse_rates": rates,
            "coarse_candidates": len(coarse),
            "validation_finalists": len(finalists),
            "selection_access": "first 60,000 shots only",
            "primary_go_condition": "absolute test error reduction >=0.003875 with positive paired block interval",
            "scientific_status": "post-baseline exploratory within a previously accessed session; requires independent confirmation",
        },
        "coarse_development": coarse,
        "validation_finalists": finalists,
        "selected": selected,
        "test": {
            "uniform_error": float(np.mean(uniform != labels[test])),
            "typed_error": float(np.mean(typed != labels[test])),
            "absolute_error_reduction": float(np.mean(uniform != labels[test]) - np.mean(typed != labels[test])),
            "relative_error_reduction": float(1.0 - np.mean(typed != labels[test]) / np.mean(uniform != labels[test])),
            "paired_block_95pct_t_interval": _interval(differences),
            "blocks": len(differences),
            "uniform_vectorized_ns_per_shot": uniform_timing["vectorized_ns_per_shot"],
            "typed_vectorized_ns_per_shot": typed_timing["vectorized_ns_per_shot"],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", required=True)
    parser.add_argument("--rates", default="0.001,0.005,0.02")
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results/rigetti_typed_matching.json"))
    args = parser.parse_args()
    payload = run(args.data, args.circuit_group, [float(value) for value in args.rates.split(",")], args.block_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": payload["selected"], "test": payload["test"]}, indent=2))
