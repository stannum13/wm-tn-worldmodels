#!/usr/bin/env python3
"""Run two cheap, directional controls before touching public tomography data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.directional import (  # noqa: E402
    fit_linear_predictor,
    generate_memory_process,
    generate_qubit_process,
    rollout_bloch_constrained,
    rollout_linear,
)


def summarize(values: list[float]) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def run(seed_count: int) -> dict[str, object]:
    memory_rows = []
    qubit_rows = []
    no_memory_rows = []
    for seed in range(seed_count):
        train = generate_memory_process(seed=seed, n_sequences=64, horizon=24)
        test = generate_memory_process(seed=10_000 + seed, n_sequences=32, horizon=60)
        markov = fit_linear_predictor(train, memory=0)
        explicit_memory = fit_linear_predictor(train, memory=1)
        memory_rows.append({
            "markov_rmse": rollout_linear(markov, test, horizon=40, memory=0),
            "memory_rmse": rollout_linear(explicit_memory, test, horizon=40, memory=1),
        })
        null_train = generate_memory_process(seed=40_000 + seed, n_sequences=64, horizon=24, memory_gain=0.0)
        null_test = generate_memory_process(seed=50_000 + seed, n_sequences=32, horizon=60, memory_gain=0.0)
        null_markov = fit_linear_predictor(null_train, memory=0)
        null_memory = fit_linear_predictor(null_train, memory=1)
        no_memory_rows.append({
            "markov_rmse": rollout_linear(null_markov, null_test, horizon=40, memory=0),
            "memory_rmse": rollout_linear(null_memory, null_test, horizon=40, memory=1),
        })

        qtrain = generate_qubit_process(seed=20_000 + seed, n_sequences=80, horizon=16)
        qtest = generate_qubit_process(seed=30_000 + seed, n_sequences=40, horizon=40, control_scale=4.0)
        qmodel = fit_linear_predictor(qtrain, memory=0)
        qraw = rollout_linear(qmodel, qtest, horizon=30, memory=0, return_states=True)
        constrained = rollout_bloch_constrained(qmodel, qtest, horizon=30)
        raw_norms = np.linalg.norm(qraw["predictions"], axis=1)
        qubit_rows.append({
            "raw_trace_distance": float(np.mean(0.5 * np.linalg.norm(qraw["predictions"] - qraw["targets"], axis=1))),
            "constrained_trace_distance": constrained["trace_distance"],
            "raw_invalid_fraction": float(np.mean(raw_norms > 1.0 + 1e-10)),
            "constrained_invalid_fraction": constrained["invalid_fraction"],
            "constrained_min_eigenvalue": constrained["min_eigenvalue"],
        })

    def mean_rows(rows: list[dict[str, float]]) -> dict[str, object]:
        return {key: summarize([row[key] for row in rows]) for key in rows[0]}

    return {
        "memory": {"per_seed": memory_rows, "mean": mean_rows(memory_rows)},
        "no_memory_null": {"per_seed": no_memory_rows, "mean": mean_rows(no_memory_rows)},
        "physicality": {"per_seed": qubit_rows, "mean": mean_rows(qubit_rows)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run(args.seeds), indent=2, sort_keys=True)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
