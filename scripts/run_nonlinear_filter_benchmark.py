#!/usr/bin/env python3
"""Compare nonlinear causal estimation accuracy with particle-filter cost."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.nonlinear_streaming import (  # noqa: E402
    particle_filter, regression_metrics, robust_ekf, simulate_nonlinear_streams,
)


def run(seeds: int, delay: int) -> dict:
    rows = []
    for seed in range(seeds):
        data = simulate_nonlinear_streams(
            seed=100_000 + seed, n_streams=30, length=800,
            artifact_probability=0.02,
        )
        estimate, timing = robust_ekf(data["observations"], delay=delay)
        rows.append({"seed": seed, "model": "robust_ekf", "particles": 0,
                     **regression_metrics(estimate, data["states"], delay=delay), **timing})
        for count in (16, 32, 64, 128):
            for contamination in (0.0, 1e-3):
                estimate, diagnostics = particle_filter(
                    data["observations"], seed=seed, particles=count, delay=delay,
                    contamination=contamination,
                )
                rows.append({
                    "seed": seed,
                    "model": "robust_particle" if contamination else "gaussian_particle",
                    "particles": count,
                    **regression_metrics(estimate, data["states"], delay=delay),
                    **diagnostics,
                })
    aggregate = []
    for key in sorted({(row["model"], row["particles"]) for row in rows}):
        group = [row for row in rows if (row["model"], row["particles"]) == key]
        numeric = [name for name in group[0] if name not in {"seed", "model", "particles"}]
        aggregate.append({"model": key[0], "particles": key[1],
                          **{name: float(np.mean([row[name] for row in group])) for name in numeric}})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "design": {"seeds": seeds, "delay": delay, "streams_per_seed": 30,
                   "length": 800, "artifact_probability": 0.02,
                   "model_access": "known dynamics and noise; privileged estimator-capacity diagnostic"},
        "per_seed": rows, "aggregate": aggregate,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/nonlinear_filter_benchmark.json"))
    args = parser.parse_args()
    payload = run(args.seeds, args.delay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["aggregate"], indent=2))
