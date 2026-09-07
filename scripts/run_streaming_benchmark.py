#!/usr/bin/env python3
"""Run the hidden-detuning streaming and delayed-response benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.streaming import benchmark_estimators, fit_gaussian_hmm, simulate_switching_streams  # noqa: E402


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "standard_error": float(np.std([row[key] for row in rows], ddof=1) / np.sqrt(len(rows))),
            "p05": float(np.quantile([row[key] for row in rows], 0.05)),
            "p95": float(np.quantile([row[key] for row in rows], 0.95)),
        }
        for key in rows[0]
    }


def run(seeds: int, train_streams: int, evaluation_streams: int, length: int) -> dict[str, object]:
    delays = [0, 5, 10, 25, 50, 100]
    raw: dict[str, dict[str, list[dict[str, float]]]] = {
        str(delay): {name: [] for name in ["instantaneous", "ewma", "hmm_filter_current", "hmm_delay_forecast"]}
        for delay in delays
    }
    fits = []
    for seed in range(seeds):
        train = simulate_switching_streams(seed=seed, n_streams=train_streams, length=length)
        fitted = fit_gaussian_hmm(train["observations"])
        evaluation = simulate_switching_streams(
            seed=100_000 + seed, n_streams=evaluation_streams, length=length
        )
        report = benchmark_estimators(evaluation, fitted, delays)
        fits.append({
            "seed": seed,
            "transition": fitted.transition.tolist(),
            "means": fitted.means.tolist(),
            "stds": fitted.stds.tolist(),
        })
        for delay, methods in report.items():
            for name, metrics in methods.items():
                raw[str(delay)][name].append(metrics)
    aggregate = {
        delay: {method: summarize(rows) for method, rows in methods.items()}
        for delay, methods in raw.items()
    }
    improvements = {}
    for delay in map(str, delays):
        current = aggregate[delay]["hmm_filter_current"]["brier_loss"]["mean"]
        predicted = aggregate[delay]["hmm_delay_forecast"]["brier_loss"]["mean"]
        best_simple = min(
            aggregate[delay][name]["brier_loss"]["mean"]
            for name in ["instantaneous", "ewma", "hmm_filter_current"]
        )
        improvements[delay] = {
            "forecast_vs_current_fraction": float((current - predicted) / current),
            "forecast_vs_best_nonforecast_fraction": float((best_simple - predicted) / best_simple),
        }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "design": {
            "seeds": seeds,
            "train_streams_per_seed": train_streams,
            "evaluation_streams_per_seed": evaluation_streams,
            "length": length,
            "delays": delays,
            "transition": [[0.995, 0.005], [0.04, 0.96]],
            "emission_means": [-0.8, 0.8],
            "emission_std": 1.0,
            "artifact_probability": 0.005,
            "dropout_probability": 0.002,
            "independent_replication_unit": "simulation seed",
        },
        "fits": fits,
        "per_seed": raw,
        "aggregate": aggregate,
        "improvements": improvements,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--train-streams", type=int, default=12)
    parser.add_argument("--evaluation-streams", type=int, default=40)
    parser.add_argument("--length", type=int, default=1200)
    parser.add_argument("--output", type=Path, default=Path("results/streaming_benchmark.json"))
    args = parser.parse_args()
    payload = run(args.seeds, args.train_streams, args.evaluation_streams, args.length)
    report = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n")
    print(report)
