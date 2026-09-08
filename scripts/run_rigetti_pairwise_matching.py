#!/usr/bin/env python3
"""Label-free pairwise-correlation graph calibration on Ankaa-2 syndromes."""

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

import h5py
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.rigetti import (  # noqa: E402
    chronological_block_error_differences,
    fit_spitz_pairwise_matching,
    load_qec_record,
    typed_circuit_noise_model,
)


def _interval(values: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def run(path: Path, train_fraction: float, block_size: int, floor_probability: float) -> dict:
    import pymatching

    with h5py.File(path, "r") as handle:
        groups = sorted(
            (int(handle[f"{name}/result"].attrs["rounds"]), name)
            for name in handle
            if name.startswith("circuit_") and "stim_circuits" in handle[name]
        )
    rows = []
    for syndrome_rounds, group in groups:
        record = load_qec_record(str(path), group)
        detectors = record["detectors"]
        labels = record["observables"][:, 0].astype(float)
        boundary = int(train_fraction * len(labels))
        train, test = np.arange(boundary), np.arange(boundary, len(labels))

        # Caune Supplementary Note 2 rates. This implementation does not yet add
        # idle-location depolarization, which is declared in the result limitation.
        template_dem = typed_circuit_noise_model(
            record["circuit"], measurement_probability=0.03,
            one_qubit_probability=0.003, two_qubit_probability=0.03,
        )
        template = pymatching.Matching.from_detector_error_model(template_dem)
        fitted, diagnostics = fit_spitz_pairwise_matching(
            template, detectors[train], floor_probability=floor_probability
        )
        baseline = np.asarray(template.decode_batch(detectors[test].astype(np.uint8)))[:, 0]
        started = perf_counter_ns()
        pairwise = np.asarray(fitted.decode_batch(detectors[test].astype(np.uint8)))[:, 0]
        pairwise_ns_per_shot = (perf_counter_ns() - started) / len(test)
        differences = chronological_block_error_differences(
            baseline, pairwise, labels[test], block_size=block_size
        )
        baseline_error = float(np.mean(baseline != labels[test]))
        pairwise_error = float(np.mean(pairwise != labels[test]))
        rows.append({
            "circuit_group": group,
            "syndrome_measurement_rounds": syndrome_rounds,
            "decoding_rounds": syndrome_rounds - 1,
            "train_rows": len(train),
            "test_rows": len(test),
            "template_error": baseline_error,
            "pairwise_error": pairwise_error,
            "absolute_error_reduction": baseline_error - pairwise_error,
            "relative_error_reduction": 1.0 - pairwise_error / baseline_error,
            "paired_row_block_95pct_t_interval": _interval(differences),
            "blocks": len(differences),
            "pairwise_vectorized_ns_per_shot": pairwise_ns_per_shot,
            "graph": diagnostics,
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
            "circuits": len(rows),
            "shots": int(sum(row["train_rows"] + row["test_rows"] for row in rows)),
        },
        "design": {
            "calibration": "label-free detector moments on the first HDF5 row range",
            "evaluation": "locked final HDF5 row range; per-shot timestamps unavailable",
            "train_fraction": train_fraction,
            "floor_probability": floor_probability,
            "pair_formula": "Spitz et al. 2018 Eq. 3 on fixed circuit-derived candidate edges",
            "primary_endpoint": "maximum decoding-round circuit",
            "primary_go_condition": ">=1% relative reduction with positive paired interval",
            "access_contract": "hard detector bits only; no logical labels or I/Q in graph fit",
            "limitation": "template follows disclosed gate rates but omits idle-location faults; not an exact reproduction of the unreleased graph implementation",
        },
        "per_circuit": rows,
        "primary": max(rows, key=lambda row: row["decoding_rounds"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--floor-probability", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("results/rigetti_pairwise_matching.json"))
    args = parser.parse_args()
    payload = run(args.data, args.train_fraction, args.block_size, args.floor_probability)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"per_circuit": payload["per_circuit"], "primary": payload["primary"]}, indent=2))
