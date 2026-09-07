#!/usr/bin/env python3
"""Run the preregistered multimodal wrapped-phase particle positive control."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.wrapped_phase import (  # noqa: E402
    grid_filter,
    phase_ekf,
    phase_metrics,
    phase_particle_filter,
    simulate_wrapped_phase,
)


def _aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for key in sorted({(row["model"], row["capacity"]) for row in rows}):
        group = [row for row in rows if (row["model"], row["capacity"]) == key]
        numeric = [name for name in group[0] if name not in {"seed", "model", "capacity"}]
        output.append({
            "model": key[0],
            "capacity": key[1],
            **{name: float(np.mean([row[name] for row in group])) for name in numeric},
        })
    return output


def _paired_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return float(np.mean(values) - half), float(np.mean(values) + half)


def run(seeds: int, streams: int, length: int, delay: int, particle_counts: list[int]) -> dict:
    rows: list[dict] = []
    for seed in range(seeds):
        data = simulate_wrapped_phase(
            seed=200_000 + seed,
            n_streams=streams,
            length=length,
            artifact_probability=0.02,
        )
        arguments = (data["observations"], data["probe_phases"])
        estimate, timing = phase_ekf(*arguments, delay=delay)
        rows.append({
            "seed": seed,
            "model": "bounded_ekf",
            "capacity": 0,
            **phase_metrics(estimate, data["states"], delay=delay),
            **timing,
        })
        for bins in (256, 512):
            estimate, diagnostics = grid_filter(*arguments, delay=delay, bins=bins)
            rows.append({
                "seed": seed,
                "model": "grid_reference",
                "capacity": bins,
                **phase_metrics(estimate, data["states"], delay=delay),
                **diagnostics,
            })
        for count in particle_counts:
            estimate, diagnostics = phase_particle_filter(
                *arguments, seed=seed, particles=count, delay=delay
            )
            rows.append({
                "seed": seed,
                "model": "robust_particle",
                "capacity": count,
                **phase_metrics(estimate, data["states"], delay=delay),
                **diagnostics,
            })

    closure = []
    for count in particle_counts:
        fractions = []
        for seed in range(seeds):
            group = [row for row in rows if row["seed"] == seed]
            loss = {(row["model"], row["capacity"]): row["circular_loss"] for row in group}
            gap = loss[("bounded_ekf", 0)] - loss[("grid_reference", 512)]
            fractions.append(
                (loss[("bounded_ekf", 0)] - loss[("robust_particle", count)]) / gap
                if gap > 0 else float("nan")
            )
        finite = np.asarray([value for value in fractions if np.isfinite(value)])
        low, high = _paired_interval(finite)
        closure.append({
            "particles": count,
            "mean_ekf_to_grid_gap_closed": float(np.mean(finite)),
            "paired_95pct_t_interval": [low, high],
            "seeds_with_positive_reference_gap": int(len(finite)),
        })
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "design": {
            "seeds": seeds,
            "streams_per_seed": streams,
            "length": length,
            "delay": delay,
            "quadrature_interval": 8,
            "artifact_probability": 0.02,
            "model_access": "known dynamics/noise; privileged nonlinear-estimator capacity diagnostic",
            "primary_endpoint": "fraction of bounded-EKF circular-loss gap to 512-bin grid closed",
            "go_condition": "mean gap closure >= 0.20 and paired interval lower endpoint > 0",
        },
        "per_seed": rows,
        "aggregate": _aggregate(rows),
        "gap_closure": closure,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--streams", type=int, default=30)
    parser.add_argument("--length", type=int, default=800)
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--particles", default="32,64,128,256")
    parser.add_argument("--output", type=Path, default=Path("results/wrapped_phase_benchmark.json"))
    args = parser.parse_args()
    payload = run(
        args.seeds,
        args.streams,
        args.length,
        args.delay,
        [int(value) for value in args.particles.split(",")],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"aggregate": payload["aggregate"], "gap_closure": payload["gap_closure"]}, indent=2))
