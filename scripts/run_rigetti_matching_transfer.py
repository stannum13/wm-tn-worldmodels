#!/usr/bin/env python3
"""Locked transfer of stability-9 matching rates to an independent acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.rigetti import (  # noqa: E402
    block_error_differences,
    load_qec_record,
    matching_predictions,
    typed_matching_predictions,
)


FROZEN_RATES = {"measurement_probability": 0.001, "one_qubit_probability": 0.005, "two_qubit_probability": 0.02}


def _interval(values: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def run(path: Path, block_size: int) -> dict:
    with h5py.File(path, "r") as handle:
        groups = []
        for name in handle:
            if name.startswith("circuit_") and "stim_circuits" in handle[name]:
                groups.append((int(handle[f"{name}/result"].attrs["rounds"]), name))
    rows = []
    block_differences: dict[str, list[float]] = {}
    for rounds, group in sorted(groups):
        record = load_qec_record(str(path), group)
        labels = record["observables"][:, 0].astype(float)
        uniform, uniform_timing = matching_predictions(record["circuit"], record["detectors"], probability=0.002)
        typed, typed_timing = typed_matching_predictions(record["circuit"], record["detectors"], **FROZEN_RATES)
        differences = block_error_differences(uniform, typed, labels, block_size=block_size)
        block_differences[group] = differences.tolist()
        rows.append({
            "circuit_group": group,
            "rounds": rounds,
            "shots": len(labels),
            "uniform_error": float(np.mean(uniform != labels)),
            "typed_error": float(np.mean(typed != labels)),
            "absolute_error_reduction": float(np.mean(uniform != labels) - np.mean(typed != labels)),
            "relative_error_reduction": float(1.0 - np.mean(typed != labels) / np.mean(uniform != labels)),
            "paired_block_95pct_t_interval": _interval(differences),
            "blocks": len(differences),
            "uniform_vectorized_ns_per_shot": uniform_timing["vectorized_ns_per_shot"],
            "typed_vectorized_ns_per_shot": typed_timing["vectorized_ns_per_shot"],
        })
    primary = max(rows, key=lambda row: row["rounds"])
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
            "independent_from_rate_selection": True,
            "circuits": len(rows),
            "shots": int(sum(row["shots"] for row in rows)),
        },
        "design": {
            "uniform_probability": 0.002,
            "frozen_typed_rates": FROZEN_RATES,
            "primary_endpoint": "maximum-round circuit logical error",
            "primary_go_condition": ">=1% relative reduction and positive paired block interval",
            "access": "no fitting or model selection on this acquisition",
        },
        "per_circuit": rows,
        "primary": primary,
        "block_differences": block_differences,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results/rigetti_matching_transfer.json"))
    args = parser.parse_args()
    payload = run(args.data, args.block_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"per_circuit": payload["per_circuit"], "primary": payload["primary"]}, indent=2))
