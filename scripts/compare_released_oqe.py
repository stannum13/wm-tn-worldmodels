#!/usr/bin/env python3
"""Compare released OQE forecasts with RB decay and a validation-gated hybrid."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.real_benchmark import fit_rb_decay, predict_rb_decay  # noqa: E402


def git_json(repo: Path, path: str):
    raw = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=repo)
    return json.loads(raw)


def arrays(payload, decay, minimum_length):
    lengths = sorted(int(k) for k in payload["labels"] if int(k) >= minimum_length)
    labels = np.concatenate([payload["labels"][str(n)] for n in lengths])
    oqe = np.concatenate([payload["predicted"][str(n)] for n in lengths])
    repeated_lengths = np.concatenate([np.full(len(payload["labels"][str(n)]), n) for n in lengths])
    rb = predict_rb_decay(decay, repeated_lengths)
    return labels, oqe, rb


def metrics(prediction, target):
    error = np.asarray(prediction) - target
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error)))}


def paired_interval(left, right, target, seed=0):
    """Bootstrap the RMSE improvement left minus right; positive favors right."""
    rng = np.random.default_rng(seed)
    n = len(target)
    values = []
    for _ in range(1000):
        ids = rng.integers(0, n, n)
        a = np.sqrt(np.mean((left[ids] - target[ids]) ** 2))
        b = np.sqrt(np.mean((right[ids] - target[ids]) ** 2))
        values.append(a - b)
    return {"mean": float(np.mean(values)), "p05": float(np.quantile(values, 0.05)), "p95": float(np.quantile(values, 0.95))}


def run(repo: Path, biases, confirmation_biases, dimension=6, gate_threshold=0.5, condition="idle100"):
    rows = []
    for bias in biases:
        base = f"experiment_data/RB_data_20230104/len40/{condition}/rb_data_{bias}"
        split = git_json(repo, base + "/data.json")
        train = split["train_data"]
        train_lengths = np.asarray([len(row[0]) for row in train])
        train_targets = np.asarray([row[1] for row in train])
        decay = fit_rb_decay(train_lengths, train_targets)

        validation = git_json(repo, base + f"/unitaryfidelities_test_40_full_D{dimension}.json")
        val_y, val_oqe, val_rb = arrays(validation, decay, 21)
        correction = val_oqe - val_rb
        denominator = float(correction @ correction)
        raw_weight = float(np.clip(correction @ (val_y - val_rb) / denominator, 0.0, 1.0)) if denominator else 0.0
        weight = raw_weight if raw_weight >= gate_threshold else 0.0

        forecast = git_json(repo, base + f"/unitaryfidelities_test_60_full_D{dimension}.json")
        y, oqe, rb = arrays(forecast, decay, 41)
        hybrid = rb + weight * (oqe - rb)
        rows.append({
            "bias": float(bias),
            "role": "confirmation" if bias in confirmation_biases else "exploratory",
            "validation_weight": raw_weight,
            "gated_weight": weight,
            "rb": metrics(rb, y),
            "oqe": metrics(oqe, y),
            "hybrid": metrics(hybrid, y),
            "hybrid_vs_rb_rmse_improvement": paired_interval(rb, hybrid, y, seed=int(float(bias) * 1000)),
            "hybrid_vs_oqe_rmse_improvement": paired_interval(oqe, hybrid, y, seed=int(float(bias) * 1000) + 1),
        })
    return {
        "protocol": {
            "oqe_dimension": dimension,
            "validation_lengths": [21, 40],
            "forecast_lengths": [41, 60],
            "gate_threshold": gate_threshold,
            "confirmation_biases": sorted(float(x) for x in confirmation_biases),
            "condition": condition,
        },
        "biases": rows,
        "mean_rmse": {
            name: float(np.mean([row[name]["rmse"] for row in rows]))
            for name in ("rb", "oqe", "hybrid")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path, help="local guochu/pt_recovery clone")
    parser.add_argument("--biases", default="0.1,0.2,0.3,0.4,0.5,0.52,0.54,0.56,0.58,0.6,0.61,0.62,0.63,0.64")
    parser.add_argument("--confirmation-biases", default="0.62,0.63,0.64")
    parser.add_argument("--dimension", type=int, default=6)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--condition", choices=("idle100", "idle180"), default="idle100")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    biases = args.biases.split(",")
    confirmation = {value for value in args.confirmation_biases.split(",") if value}
    report = run(args.repo, biases, confirmation, args.dimension, args.gate_threshold, args.condition)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
